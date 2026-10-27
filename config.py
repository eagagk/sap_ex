import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson",
)

IMAGE_INDEX_PATH = os.path.join(BASE_DIR, "image_hnsw_all.bin")
TEXT_INDEX_PATH = os.path.join(BASE_DIR, "text_hnsw_all.bin")
METADATA_PATH = os.path.join(BASE_DIR, "product_metadata_all.json")

MODEL_ID = "patrickjohncyh/fashion-clip"

WEIGHTS = {
    "image": 0.40,
    "text": 0.20,
    "price": 0.20,
    "weight": 0.10,
    "rating": 0.10,
}

SCALING_FACTORS = {
    "price_diff": 50.0,
    "weight_diff": 100.0,
    "rating_diff": 1.0,
}

MAX_RECOMMENDATIONS = 50
DEFAULT_CANDIDATE_POOL = 150
MAX_IMAGES_PER_PRODUCT = 5