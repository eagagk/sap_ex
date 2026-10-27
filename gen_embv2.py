import json
from io import BytesIO

import faiss
import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

import config
from tqdm.auto import tqdm
from utils import clean_weight, get_all_valid_urls, to_float


IMG_VECTOR_DIM = 768
TXT_VECTOR_DIM = 512

M_LINKS = 32
EF_CONSTRUCTION = 100
EF_SEARCH = 50


def safe_text(value):
    """Convert dataset values safely into usable text."""
    if value is None:
        return ""

    if isinstance(value, float) and np.isnan(value):
        return ""

    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)

    value = str(value).strip()

    if value.lower() in {"", "nan", "none", "null"}:
        return ""

    return value


def build_product_text(row):
    """Create one rich text description per product."""
    # fields = {
    #     "Product": row.get("product_name"),
    #     "Brand": row.get("brand"),
    #     "Color": row.get("colour"),
    #     "Category": row.get("parent___child_category__all"),
    #     "Keywords": row.get("meta_keywords"),
    #     "Details": row.get("product_details__k_v_pairs"),
    #     "Technical details": row.get("technical_details__k_v_pairs"),
    # }
    fields = {
        "Product": row.get("product_name"),
        "Brand": row.get("brand"),
        "Color": row.get("colour"),
        "Category": row.get("parent___child_category__all"),
        "Keywords": row.get("meta_keywords"),
    }

    parts = []

    for label, value in fields.items():
        text = safe_text(value)
        if text:
            parts.append(f"{label}: {text}")

    return ". ".join(parts) or "Fashion product"


def get_product_image_urls(row):
    """
    Return all image views for a product.
    Prefer large images, then medium, then small.
    """
    for column in ("large", "medium", "image_urls__small"):
        urls = get_all_valid_urls(safe_text(row.get(column)))
        if urls:
            return urls

    return []


def generate_single_image_embedding(image_url, model, processor):
    """Generate one normalized visual embedding."""
    try:
        response = requests.get(image_url, timeout=15)
        response.raise_for_status()

        image = Image.open(BytesIO(response.content)).convert("RGB")

        inputs = processor(images=[image], return_tensors="pt")

        with torch.inference_mode():
            features = model.vision_model(**inputs).pooler_output
            features = features / features.norm(dim=-1, keepdim=True)

        return features.squeeze(0).cpu().numpy().astype("float32")

    except Exception as error:
        print(f"[ERROR] Image embedding failed for {image_url}: {error}")
        return None


def generate_text_embedding(text_string, model, processor):
    """Generate one normalized text embedding."""
    try:
        max_length = model.config.text_config.max_position_embeddings

        inputs = processor(
            text=[text_string],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )

        with torch.inference_mode():
            features = model.text_model(**inputs).pooler_output
            features = features / features.norm(dim=-1, keepdim=True)

        return features.squeeze(0).cpu().numpy().astype("float32")

    except Exception as error:
        print(f"[ERROR] Text embedding failed: {error}")
        return None


def generate_average_image_embedding(image_urls, model, processor):
    """
    Create one product-level visual vector from multiple orientations.

    1. Embed each product image.
    2. Mean-pool all successful embeddings.
    3. Normalize the pooled vector.
    """
    embeddings = []

    for image_url in image_urls[:config.MAX_IMAGES_PER_PRODUCT]:
        embedding = generate_single_image_embedding(
            image_url=image_url,
            model=model,
            processor=processor,
        )

        if embedding is not None:
            embeddings.append(embedding)

    if not embeddings:
        return None

    pooled_embedding = np.mean(
        np.vstack(embeddings),
        axis=0,
    ).astype("float32")

    norm = np.linalg.norm(pooled_embedding)

    if norm == 0:
        return None

    return pooled_embedding / norm


def index_embedding(embedding, faiss_id, index):
    """Add one vector to a FAISS index with its product FAISS ID."""
    vector = embedding.reshape(1, -1).astype("float32")
    ids = np.array([faiss_id], dtype="int64")
    index.add_with_ids(vector, ids)


def run_batch_indexing_pipeline(limit=None):
    print(f"Reading product catalog: {config.DATA_PATH}")
    dataframe = pd.read_json(config.DATA_PATH, lines=True)

    if limit is not None:
        dataframe = dataframe.head(limit)

    print(f"Loading model: {config.MODEL_ID}")
    model = CLIPModel.from_pretrained(config.MODEL_ID)
    processor = CLIPProcessor.from_pretrained(config.MODEL_ID)
    model.eval()

    image_base_index = faiss.IndexHNSWFlat(
        IMG_VECTOR_DIM,
        M_LINKS,
        faiss.METRIC_INNER_PRODUCT,
    )

    text_base_index = faiss.IndexHNSWFlat(
        TXT_VECTOR_DIM,
        M_LINKS,
        faiss.METRIC_INNER_PRODUCT,
    )

    image_base_index.hnsw.efConstruction = EF_CONSTRUCTION
    image_base_index.hnsw.efSearch = EF_SEARCH

    text_base_index.hnsw.efConstruction = EF_CONSTRUCTION
    text_base_index.hnsw.efSearch = EF_SEARCH

    image_index = faiss.IndexIDMap2(image_base_index)
    text_index = faiss.IndexIDMap2(text_base_index)

    metadata = {}
    indexed = 0
    skipped = 0

    progress_bar = tqdm(
        dataframe.iterrows(),
        total=len(dataframe),
        desc="Indexing products",
        unit="product",
    )

    for _, row in progress_bar:
        image_urls = get_product_image_urls(row)

        # Strategy 1: exclude products that have no valid image URLs
        # from image index, text index, and metadata.
        if not image_urls:
            skipped += 1
            progress_bar.set_postfix(
                indexed=indexed,
                skipped=skipped,
                status="no image",
            )
            continue

        product_text = build_product_text(row)

        # Average-pool all valid product-image embeddings.
        image_embedding = generate_average_image_embedding(
            image_urls=image_urls,
            model=model,
            processor=processor,
        )

        # Generate one text vector for the product description.
        text_embedding = generate_text_embedding(
            text_string=product_text,
            model=model,
            processor=processor,
        )

        # Skip product if either representation could not be produced.
        if image_embedding is None or text_embedding is None:
            skipped += 1
            progress_bar.set_postfix(
                indexed=indexed,
                skipped=skipped,
                status="embedding failed",
            )
            continue

        faiss_id = indexed

        index_embedding(
            embedding=image_embedding,
            faiss_id=faiss_id,
            index=image_index,
        )

        index_embedding(
            embedding=text_embedding,
            faiss_id=faiss_id,
            index=text_index,
        )

        metadata[faiss_id] = {
            "sku": safe_text(row.get("uniq_id")),
            "name": safe_text(row.get("product_name")),
            "brand": safe_text(row.get("brand")),
            "colour": safe_text(row.get("colour")),
            "category": safe_text(row.get("parent___child_category__all")),
            "price": to_float(row.get("sales_price")),
            "weight": clean_weight(row.get("weight")),
            "rating": to_float(row.get("rating")),
            "review_count": to_float(row.get("no__of_reviews")),
            "discount_percentage": to_float(row.get("discount_percentage")),
            "image_count": len(image_urls),
            "search_text": product_text,
            "media": {
                "large": safe_text(row.get("large")),
                "medium": safe_text(row.get("medium")),
                "small": safe_text(row.get("image_urls__small")),
            },
        }

        indexed += 1

        progress_bar.set_postfix(
            indexed=indexed,
            skipped=skipped,
            images_used=min(
                len(image_urls),
                config.MAX_IMAGES_PER_PRODUCT,
            ),
            status="indexed",
        )

    progress_bar.close()

    if indexed == 0:
        raise RuntimeError(
            "No products were indexed. Check dataset fields and image URLs."
        )

    print("\nWriting FAISS indexes...")
    faiss.write_index(image_index, config.IMAGE_INDEX_PATH)
    faiss.write_index(text_index, config.TEXT_INDEX_PATH)

    print("Writing product metadata...")
    with open(config.METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, allow_nan=False)

    print("\nIndexing completed successfully.")
    print(f"Indexed products : {indexed}")
    print(f"Skipped products : {skipped}")
    print(f"Image index      : {config.IMAGE_INDEX_PATH}")
    print(f"Text index       : {config.TEXT_INDEX_PATH}")
    print(f"Metadata file    : {config.METADATA_PATH}")


if __name__ == "__main__":
    run_batch_indexing_pipeline()