import numpy as np
import pandas as pd

def _safe_float(value, default=0.0):

    try:
        number = float(value)

    except (TypeError, ValueError):
        return default

    if pd.isna(number):
        return default

    return number


def _clip01(value):

    return float(
        np.clip(
            _safe_float(value),
            0.0,
            1.0
        )
    )