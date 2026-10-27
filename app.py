from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException, Query

import config
from sim_search import ProductSearchEngine
from utils import clean_weight, get_first_valid_url, to_float

search_engine = ProductSearchEngine()


@asynccontextmanager
async def lifespan(app):
    search_engine.load()
    print(
        f"Recommendation API ready: "
        f"{len(search_engine.metadata_by_id)} products loaded."
    )
    yield


app = FastAPI(
    title="Fashion Recommendation API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "indexed_products": len(search_engine.metadata_by_id),
    }


@app.get("/recommend")
def find_recommended_products(
    product_id: str = Query(..., min_length=1),
    num_items: int = Query(
        default=5,
        ge=1,
        le=config.MAX_RECOMMENDATIONS,
        description="Number of recommendations to return",
    ),
):
    recommendations = search_engine.recommend(
        sku=product_id,
        num_items=num_items,
    )

    if recommendations is None:
        raise HTTPException(
            status_code=404,
            detail=f"Product '{product_id}' was not found in the indexed catalog.",
        )

    return {
        "searched_for": product_id,
        "requested_count": num_items,
        "returned_count": len(recommendations),
        "recommendations": recommendations,
    }

@app.get(
    "/find_similar_products",
    response_model=List[str],
)
def find_similar_products(
    product_id: str = Query(..., min_length=1),
    num_similar: int = Query(
        5,
        ge=1,
        le=config.MAX_RECOMMENDATIONS,
    ),
) -> List[str]:
    try:
        recommendations = search_engine.recommend(
            sku=product_id,
            num_items=num_similar,
        )

        if recommendations is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Product '{product_id}' was not found "
                    "in the catalog."
                ),
            )

        return [
            recommendation["product_id"]
            for recommendation in recommendations
        ]

    except HTTPException:
        raise

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except Exception as error:
        print(f"Recommendation error: {error}")

        raise HTTPException(
            status_code=500,
            detail="Could not generate recommendations.",
        ) from error

@app.get("/products/{product_id}")
def get_product_details(product_id: str):
    """
    Return details of the requested source-dataset product only.
    Does not run recommendation search.
    """
    product = search_engine.catalog_by_sku.get(product_id)

    if product is None:
        raise HTTPException(
            status_code=404,
            detail=f"Product '{product_id}' was not found in the dataset.",
        )

    colour = str(product.get("colour", "")).strip()

    if colour.lower() in {"", "nan", "none", "null"}:
        colour = None

    name = str(product.get("product_name", "")).strip()

    if name.lower() in {"", "nan", "none", "null"}:
        name = None

    brand = str(product.get("brand", "")).strip()

    if brand.lower() in {"", "nan", "none", "null"}:
        brand = None

    return {
        "id": str(product.get("uniq_id")),
        "name": name,
        "image_url_large": get_first_valid_url(
            str(product.get("large", ""))
        ),
        "brand": brand,
        "price": to_float(product.get("sales_price")),
        "color": colour,
        "weight_g": clean_weight(product.get("weight")),
    }