import re
import numpy as np


def to_float(value, default=None):
    """Convert a value to float safely; return default for invalid values."""
    if value is None:
        return default

    try:
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return default
        return value
    except (TypeError, ValueError):
        return default


def clean_weight(weight_value):
    """
    Convert values such as '500g', '1.2 kg', and '2 lbs' to grams.
    Unknown/sentinel values return None.
    """
    if weight_value is None:
        return None

    weight_str = str(weight_value).strip().lower()
    if not weight_str or weight_str in {"nan", "none", "null"}:
        return None

    matches = re.findall(r"[-+]?\d*\.\d+|\d+", weight_str)
    if not matches:
        return None

    value = float(matches[0])

    # Dataset uses values such as 999999999 as an unknown-weight sentinel.
    if value >= 99_999_999:
        return None

    if "kg" in weight_str:
        return value * 1000
    if "lb" in weight_str or "pound" in weight_str:
        return value * 453.592

    return value


def calculate_proximity(val1, val2, scale):
    """Return an exponential similarity score from 0.0 to 1.0."""
    val1 = to_float(val1)
    val2 = to_float(val2)

    if val1 is None or val2 is None:
        return 0.5

    return float(np.exp(-abs(val1 - val2) / scale))


def get_first_valid_url(url_field):
    """Return the first HTTP URL from a pipe-separated field."""
    if not isinstance(url_field, str):
        return ""

    urls = [
        url.strip()
        for url in url_field.split("|")
        if url.strip().startswith(("http://", "https://"))
    ]
    return urls[0] if urls else ""


def get_all_valid_urls(url_field):
    """Return all HTTP URLs from a pipe-separated field."""
    if not isinstance(url_field, str):
        return []

    return [
        url.strip()
        for url in url_field.split("|")
        if url.strip().startswith(("http://", "https://"))
    ]


def get_best_image_url(row):
    """Return the first usable image URL, preferring large images."""
    for column in ("large", "medium", "image_urls__small"):
        url = get_first_valid_url(str(row.get(column, "")))
        if url:
            return url
    return ""