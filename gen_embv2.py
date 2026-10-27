import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from itertools import islice

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
BATCH_SIZE = 32
IMAGE_INFERENCE_BATCH_SIZE = 32
MAX_DOWNLOAD_WORKERS = 16


def safe_text(value):
    """Convert dataset values safely into usable text."""
    if value is None:
        return ""

    if isinstance(value, float) and np.isnan(value):
        return ""

    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)

    value = str(value).strip()

    if value.lower() in {"", "nan", "none", "null", "<na>"}:
        return ""

    return value


def build_product_text(row):
    """Create one rich text description per product."""
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

    return ". ".join(parts) if parts else None


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


def download_image(image_url):
    """Download one product image for the current batch."""
    try:
        response = requests.get(
            image_url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")

    except Exception as error:
        print(f"[WARN] Image download failed for {image_url}: {error}")
        return None


def download_images_parallel(image_urls):
    """Download unique image URLs concurrently for one product batch."""
    unique_urls = list(dict.fromkeys(image_urls))

    if not unique_urls:
        return {}

    downloaded = {}
    worker_count = min(MAX_DOWNLOAD_WORKERS, len(unique_urls))

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_url = {
            executor.submit(download_image, image_url): image_url
            for image_url in unique_urls
        }

        for future in as_completed(future_to_url):
            image_url = future_to_url[future]
            downloaded[image_url] = future.result()

    return downloaded


def _encode_image_batch(images, model, processor, device):
    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.inference_mode():
        features = model.vision_model(
            pixel_values=pixel_values
        ).pooler_output
        features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy().astype("float32")


def generate_batch_image_embeddings(
    batch_url_lists,
    image_cache,
    model,
    processor,
    device,
):
    """Batch image inference and mean-pool each product's image views."""
    flat_images = []
    flat_product_indices = []

    for product_index, image_urls in enumerate(batch_url_lists):
        for image_url in image_urls[:config.MAX_IMAGES_PER_PRODUCT]:
            image = image_cache.get(image_url)
            if image is not None:
                flat_images.append(image)
                flat_product_indices.append(product_index)

    if not flat_images:
        return [None] * len(batch_url_lists)

    image_embeddings = [None] * len(flat_images)

    for start in range(0, len(flat_images), IMAGE_INFERENCE_BATCH_SIZE):
        end = min(start + IMAGE_INFERENCE_BATCH_SIZE, len(flat_images))

        try:
            batch_embeddings = _encode_image_batch(
                flat_images[start:end],
                model,
                processor,
                device,
            )
            image_embeddings[start:end] = list(batch_embeddings)
        except Exception as error:
            print(f"[WARN] Image batch inference failed: {error}")

            # Preserve per-image failure isolation if a whole batch fails.
            for image_index in range(start, end):
                try:
                    image_embeddings[image_index] = _encode_image_batch(
                        [flat_images[image_index]],
                        model,
                        processor,
                        device,
                    )[0]
                except Exception as image_error:
                    print(f"[WARN] Image inference failed: {image_error}")

    product_embeddings = [[] for _ in batch_url_lists]

    for embedding, product_index in zip(
        image_embeddings,
        flat_product_indices,
    ):
        if embedding is not None:
            product_embeddings[product_index].append(embedding)

    pooled_embeddings = []

    for embeddings in product_embeddings:
        if not embeddings:
            pooled_embeddings.append(None)
            continue

        pooled = np.mean(
            np.vstack(embeddings),
            axis=0,
        ).astype("float32")
        norm = np.linalg.norm(pooled)
        pooled_embeddings.append(pooled / norm if norm > 0 else None)

    return pooled_embeddings


def _encode_text_batch(texts, model, processor, device):
    max_length = model.config.text_config.max_position_embeddings
    inputs = processor(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        features = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        ).pooler_output
        features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy().astype("float32")


def generate_batch_text_embeddings(texts, model, processor, device):
    """Batch text inference while preserving missing-text positions."""
    embeddings = [None] * len(texts)
    valid_items = [
        (index, text)
        for index, text in enumerate(texts)
        if text
    ]

    for start in range(0, len(valid_items), BATCH_SIZE):
        batch_items = valid_items[start:start + BATCH_SIZE]
        batch_texts = [text for _, text in batch_items]

        try:
            batch_embeddings = _encode_text_batch(
                batch_texts,
                model,
                processor,
                device,
            )
            for (index, _), embedding in zip(
                batch_items,
                batch_embeddings,
            ):
                embeddings[index] = embedding
        except Exception as error:
            print(f"[WARN] Text batch inference failed: {error}")

            # Preserve per-product failure isolation if a whole batch fails.
            for index, text in batch_items:
                try:
                    embeddings[index] = _encode_text_batch(
                        [text],
                        model,
                        processor,
                        device,
                    )[0]
                except Exception as text_error:
                    print(f"[WARN] Text inference failed: {text_error}")

    return embeddings


def iter_batches(iterable, batch_size):
    iterator = iter(iterable)

    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            return
        yield batch


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model: {config.MODEL_ID} on {device}")
    model = CLIPModel.from_pretrained(config.MODEL_ID).to(device)
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
        total=len(dataframe),
        desc="Indexing products",
        unit="product",
    )

    for row_batch in iter_batches(dataframe.iterrows(), BATCH_SIZE):
        records = []

        for _, row in row_batch:
            sku = safe_text(row.get("uniq_id"))

            if not sku:
                skipped += 1
                continue

            image_urls = get_product_image_urls(row)
            product_text = build_product_text(row)
            price = to_float(row.get("sales_price"))
            weight = clean_weight(row.get("weight"))
            rating = to_float(row.get("rating"))

            has_numeric_data = any(
                value is not None
                for value in (price, weight, rating)
            )

            if not image_urls and not product_text and not has_numeric_data:
                skipped += 1
                continue

            records.append({
                "row": row,
                "sku": sku,
                "image_urls": image_urls,
                "product_text": product_text,
                "price": price,
                "weight": weight,
                "rating": rating,
            })

        batch_url_lists = [record["image_urls"] for record in records]
        all_image_urls = [
            image_url
            for image_urls in batch_url_lists
            for image_url in image_urls[:config.MAX_IMAGES_PER_PRODUCT]
        ]
        image_cache = download_images_parallel(all_image_urls)

        image_embeddings = generate_batch_image_embeddings(
            batch_url_lists=batch_url_lists,
            image_cache=image_cache,
            model=model,
            processor=processor,
            device=device,
        )
        text_embeddings = generate_batch_text_embeddings(
            texts=[record["product_text"] for record in records],
            model=model,
            processor=processor,
            device=device,
        )

        for record, image_embedding, text_embedding in zip(
            records,
            image_embeddings,
            text_embeddings,
        ):
            has_numeric_data = any(
                value is not None
                for value in (
                    record["price"],
                    record["weight"],
                    record["rating"],
                )
            )

            if (
                image_embedding is None
                and text_embedding is None
                and not has_numeric_data
            ):
                skipped += 1
                continue

            faiss_id = indexed

            if image_embedding is not None:
                index_embedding(image_embedding, faiss_id, image_index)

            if text_embedding is not None:
                index_embedding(text_embedding, faiss_id, text_index)

            row = record["row"]
            metadata[faiss_id] = {
                "sku": record["sku"],
                "name": safe_text(row.get("product_name")),
                "brand": safe_text(row.get("brand")),
                "colour": safe_text(row.get("colour")),
                "category": safe_text(row.get("parent___child_category__all")),
                "price": record["price"],
                "weight": record["weight"],
                "rating": record["rating"],
                "has_image": image_embedding is not None,
                "has_text": text_embedding is not None,
                "review_count": to_float(row.get("no__of_reviews")),
                "discount_percentage": to_float(row.get("discount_percentage")),
                "image_count": len(record["image_urls"]),
                "search_text": record["product_text"] or "",
                "media": {
                    "large": safe_text(row.get("large")),
                    "medium": safe_text(row.get("medium")),
                    "small": safe_text(row.get("image_urls__small")),
                },
            }
            indexed += 1

        progress_bar.update(len(row_batch))
        progress_bar.set_postfix(indexed=indexed, skipped=skipped)

        del image_cache

    progress_bar.close()

    if indexed == 0:
        raise RuntimeError(
            "No products had a valid ID and usable image, text, or numeric features."
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