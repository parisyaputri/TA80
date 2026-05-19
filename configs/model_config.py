# configs/model_config.py

class DDCConfig:

    Z_THRESHOLD = 2.5
    IQR_MULTIPLIER = 2.0
    EWMA_MARGIN = 0.10


class ARMConfig:

    MIN_SUPPORT = 0.06
    MIN_CONFIDENCE = 0.65
    MIN_LIFT = 1.0
    MAX_RULES = 40