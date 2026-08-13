"""AQI category mapping.

Extracted from the original ``data/download.py`` so that the model and
evaluation layers do not need to import the data-download pipeline.
"""

import pandas as pd

from src.aq_forecast.config import AQI_CATEGORIES


def get_aqi_category(aqi_value: float) -> str:
    """Map a numerical AQI value to its categorical label."""
    if pd.isna(aqi_value):
        return "Unknown"
    for category, (low, high) in AQI_CATEGORIES.items():
        if low <= aqi_value <= high:
            return category
    return "Severe" if aqi_value > 500 else "Unknown"
