import json
import os
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import faiss
import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

import config
from utils import (
    calculate_proximity,
    clean_weight,
    get_all_valid_urls,
    get_first_valid_url,
    to_float,
)


class ProductSearchEngine:
    def __init__(self):
        self.image_index = None
        self.text_index = None

        self.metadata_by_id = {}
        self.sku_to_faiss_id = {}

        # Raw source dataset, used for every incoming query.
        self.catalog_by_sku = {}

        self.model = None
        self.processor = None
        self.device = None

    def load(self):
        required_files = [
            config.DATA_PATH,
            config.IMAGE_INDEX_PATH,
            config.TEXT_INDEX_PATH,
            config.METADATA_PATH,
        ]

        missing = [
            file_path
            for file_path in required_files
            if not os.path.exists(file_path)
        ]

        if missing:
            raise RuntimeError(
                "Required files are missing: "
                + ", ".join(missing)
            )

        # Load candidate indexes.
        self.image_index = faiss.read_index(
            config.IMAGE_INDEX_PATH
        )

        self.text_index = faiss.read_index(
            config.TEXT_INDEX_PATH
        )

        # Load candidate metadata.
        with open(
            config.METADATA_PATH,
            "r",
            encoding="utf-8",
        ) as file:
            raw_metadata = json.load(file)

        self.metadata_by_id = {
            int(faiss_id): metadata
            for faiss_id, metadata in raw_metadata.items()
        }

        self.sku_to_faiss_id = {
            metadata["sku"]: faiss_id
            for faiss_id, metadata in self.metadata_by_id.items()
        }

        # This implementation expects every indexed candidate
        # to have both image and text vectors.
        if self.image_index.ntotal != self.text_index.ntotal:
            raise RuntimeError(
                "Image and text index counts differ. "
                "Rebuild both indexes together."
            )

        if self.image_index.ntotal != len(self.metadata_by_id):
            raise RuntimeError(
                "Index count and metadata count differ. "
                "Rebuild all assets together."
            )

        # Load the source catalog.
        dataframe = pd.read_json(
            config.DATA_PATH,
            lines=True,
        )

        self.catalog_by_sku = {
            str(row["uniq_id"]): row.to_dict()
            for _, row in dataframe.iterrows()
            if pd.notna(row.get("uniq_id"))
        }

        # Load FashionCLIP for fresh query embedding generation.
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Loading FashionCLIP on: {self.device}")

        self.model = CLIPModel.from_pretrained(
            config.MODEL_ID
        ).to(self.device)

        self.processor = CLIPProcessor.from_pretrained(
            config.MODEL_ID
        )

        self.model.eval()

        # HNSW query recall setting.
        image_base = getattr(
            self.image_index,
            "index",
            self.image_index,
        )

        text_base = getattr(
            self.text_index,
            "index",
            self.text_index,
        )

        if hasattr(image_base, "hnsw"):
            image_base.hnsw.efSearch = 50

        if hasattr(text_base, "hnsw"):
            text_base.hnsw.efSearch = 50

        print(
            f"Loaded {len(self.catalog_by_sku)} source products, "
            f"{self.image_index.ntotal} indexed products."
        )

    @staticmethod
    def _safe_text(value):
        if value is None:
            return ""

        if isinstance(value, float) and np.isnan(value):
            return ""

        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)

        value = str(value).strip()

        if value.lower() in {
            "",
            "nan",
            "none",
            "null",
            "<na>",
        }:
            return ""

        return value

    def _build_product_text(self, row):
        """
        Keep this text-building logic identical to gen_emd.py.
        The query text must use the same format as indexed text.
        """
        fields = {
            "Product": row.get("product_name"),
            "Brand": row.get("brand"),
            "Color": row.get("colour"),
            "Category": row.get(
                "parent___child_category__all"
            ),
            "Keywords": row.get("meta_keywords"),
        }

        parts = []

        for label, value in fields.items():
            text = self._safe_text(value)

            if text:
                parts.append(f"{label}: {text}")

        return ". ".join(parts) or "Fashion product"

    def _get_product_image_urls(self, row):
        """Return image views, preferring large images."""
        for column in (
            "large",
            "medium",
            "image_urls__small",
        ):
            urls = get_all_valid_urls(
                self._safe_text(row.get(column))
            )

            if urls:
                return urls[:config.MAX_IMAGES_PER_PRODUCT]

        return []

    @staticmethod
    def _download_image(image_url):
        """Download one product image."""
        try:
            response = requests.get(
                image_url,
                timeout=(5, 20),
                headers={
                    "User-Agent": "Mozilla/5.0",
                },
            )

            response.raise_for_status()

            with Image.open(
                BytesIO(response.content)
            ) as image:
                return image.convert("RGB")

        except Exception:
            return None

    def _generate_image_embedding(self, image_urls):
        """
        Generate a new 768-dimensional pooled image embedding
        from all available image orientations.
        """
        images = []

        for image_url in image_urls:
            image = self._download_image(image_url)

            if image is not None:
                images.append(image)

        if not images:
            return None

        inputs = self.processor(
            images=images,
            return_tensors="pt",
        )

        pixel_values = inputs["pixel_values"].to(
            self.device
        )

        with torch.inference_mode():
            vectors = self.model.vision_model(
                pixel_values=pixel_values
            ).pooler_output

            vectors = vectors / vectors.norm(
                dim=-1,
                keepdim=True,
            )

        vectors = vectors.cpu().numpy().astype("float32")

        # Average-pool all image orientations for one product vector.
        pooled_vector = np.mean(
            vectors,
            axis=0,
        ).astype("float32")

        norm = np.linalg.norm(pooled_vector)

        if norm == 0:
            return None

        return pooled_vector / norm

    def _generate_text_embedding(self, product_text):
        """
        Generate a new normalized 512-dimensional text embedding.
        """
        max_length = (
            self.model.config.text_config.max_position_embeddings
        )

        inputs = self.processor.tokenizer(
            product_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            vector = self.model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            ).pooler_output

            vector = vector / vector.norm(
                dim=-1,
                keepdim=True,
            )

        return vector.squeeze(0).cpu().numpy().astype(
            "float32"
        )

    @staticmethod
    def _normalize_inner_product(score):
        """
        Indexed vectors are normalized, so inner product is cosine
        similarity in [-1, 1]. Convert it into [0, 1].
        """
        return float(
            np.clip((score + 1.0) / 2.0, 0.0, 1.0)
        )

    def recommend(self, sku, num_items=5):
        """
        Every request generates fresh FashionCLIP query embeddings.

        The query SKU must exist in the source dataset.
        It does not need to be present in the FAISS index.
        """
        query_row = self.catalog_by_sku.get(sku)

        if query_row is None:
            return None

        query_image_urls = self._get_product_image_urls(
            query_row
        )

        query_text = self._build_product_text(query_row)

        # Step 1: Generate fresh query embeddings.
        query_image_embedding = self._generate_image_embedding(
            query_image_urls
        )

        query_text_embedding = self._generate_text_embedding(
            query_text
        )

        if query_image_embedding is None:
            raise ValueError(
                f"Could not create an image embedding for '{sku}'."
            )

        if query_text_embedding is None:
            raise ValueError(
                f"Could not create a text embedding for '{sku}'."
            )

        query_image_vector = query_image_embedding.reshape(
            1,
            -1,
        )

        query_text_vector = query_text_embedding.reshape(
            1,
            -1,
        )

        candidate_depth = min(
            config.DEFAULT_CANDIDATE_POOL,
            self.image_index.ntotal,
            self.text_index.ntotal,
        )

        # Step 2: Image and text FAISS searches run independently.
        with ThreadPoolExecutor(max_workers=2) as executor:
            image_future = executor.submit(
                self.image_index.search,
                query_image_vector,
                candidate_depth,
            )

            text_future = executor.submit(
                self.text_index.search,
                query_text_vector,
                candidate_depth,
            )

            image_scores, image_ids = image_future.result()
            text_scores, text_ids = text_future.result()

        image_score_map = {
            int(faiss_id): self._normalize_inner_product(
                float(score)
            )
            for faiss_id, score in zip(
                image_ids[0],
                image_scores[0],
            )
            if faiss_id != -1
        }

        text_score_map = {
            int(faiss_id): self._normalize_inner_product(
                float(score)
            )
            for faiss_id, score in zip(
                text_ids[0],
                text_scores[0],
            )
            if faiss_id != -1
        }

        # Step 3: Union candidate IDs from both FAISS searches.
        candidate_ids = (
            set(image_score_map)
            .union(text_score_map)
        )

        # Remove source product if it already exists in the index.
        query_faiss_id = self.sku_to_faiss_id.get(sku)

        if query_faiss_id is not None:
            candidate_ids.discard(query_faiss_id)

        query_price = to_float(
            query_row.get("sales_price")
        )

        query_weight = clean_weight(
            query_row.get("weight")
        )

        query_rating = to_float(
            query_row.get("rating")
        )

        results = []

        # Step 4: Calculate hybrid score for every union candidate.
        for candidate_id in candidate_ids:
            candidate_meta = self.metadata_by_id.get(
                candidate_id
            )

            if candidate_meta is None:
                continue

            # Candidate may be retrieved by only one modality.
            # No FAISS reconstruction is performed.
            image_similarity = image_score_map.get(
                candidate_id,
                0.0,
            )

            text_similarity = text_score_map.get(
                candidate_id,
                0.0,
            )

            price_similarity = calculate_proximity(
                candidate_meta.get("price"),
                query_price,
                config.SCALING_FACTORS["price_diff"],
            )

            weight_similarity = calculate_proximity(
                candidate_meta.get("weight"),
                query_weight,
                config.SCALING_FACTORS["weight_diff"],
            )

            rating_similarity = calculate_proximity(
                candidate_meta.get("rating"),
                query_rating,
                config.SCALING_FACTORS["rating_diff"],
            )

            final_score = (
                image_similarity * config.WEIGHTS["image"]
                + text_similarity * config.WEIGHTS["text"]
                + price_similarity * config.WEIGHTS["price"]
                + weight_similarity * config.WEIGHTS["weight"]
                + rating_similarity * config.WEIGHTS["rating"]
            )

            media = candidate_meta.get("media", {})

            results.append({
                "product_id": candidate_meta["sku"],
                "final_match_score": round(
                    float(final_score),
                    6,
                ),
                "score_breakdown": {
                    "image": round(image_similarity, 6),
                    "text": round(text_similarity, 6),
                    "price_similarity": round(
                        price_similarity,
                        6,
                    ),
                    "weight_similarity": round(
                        weight_similarity,
                        6,
                    ),
                    "rating_similarity": round(
                        rating_similarity,
                        6,
                    ),
                },
                "media": {
                    "image_url": (
                        get_first_valid_url(
                            media.get("large", "")
                        )
                        or get_first_valid_url(
                            media.get("medium", "")
                        )
                        or get_first_valid_url(
                            media.get("small", "")
                        )
                    ),
                },
                "attributes": {
                    "sales_price": to_float(
                        candidate_meta.get("price")
                    ),
                    "rating": to_float(
                        candidate_meta.get("rating")
                    ),
                    "weight_g": clean_weight(
                        candidate_meta.get("weight")
                    ),
                },
            })

        # Step 5: Sort descending and return requested number.
        results.sort(
            key=lambda result: result["final_match_score"],
            reverse=True,
        )

        return results[:num_items]