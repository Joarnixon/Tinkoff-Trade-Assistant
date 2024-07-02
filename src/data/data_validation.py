import polars as pl
from typing import Optional

def validate_online(data: pl.DataFrame) -> Optional[pl.DataFrame]:
    """
    For online data stream that comes in a form of single row. Return None if there are any null values.
    """
    if sum(data.fill_nan(None).null_count()).item() > 0:
        return None
    return data

def validate(data: pl.DataFrame) -> pl.DataFrame:
    """
    Handles nan values.
    """
    data = data.fill_nan(None)

    if sum(data.null_count()).item() > 0:
        data = data.interpolate()
        if sum(data.null_count()).item() > 0:
            data = data.fill_null(strategy="forward")
            data = data.fill_null(strategy="backward")
    return data