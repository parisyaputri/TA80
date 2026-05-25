# configs/model_config.py

class DDCConfig:
    # Thresholds for statistical outliers
    Z_THRESHOLD = 2.5
    IQR_MULTIPLIER = 2.0
    EWMA_MARGIN = 0.10
    
    # Severity calculation normalization factors
    SEVERITY_Z_DIVISOR = 3.0
    SEVERITY_VIOL_DIVISOR = 4.0

    # Hard rules bounds
    ALLOWED_MISSING_STEPS_MAX = 0
    ALLOWED_DUPLICATE_STEPS_MAX = 2
    ALLOWED_SEQ_VIOLATIONS_MAX = 1
    ALLOWED_UNUSUAL_RESOURCE_EVENTS_MAX = 0

    # EWMA settings
    EWMA_ALPHA = 0.20



class ARMConfig:
    # Association Rule Mining parameters
    MIN_SUPPORT = 0.06
    MIN_CONFIDENCE = 0.65
    MIN_LIFT = 1.05
    MAX_RULES = 40
    
    # Severity weighting for ARM rule hits
    CONFIDENCE_WEIGHT = 0.50
    LIFT_WEIGHT = 0.30
    SUPPORT_WEIGHT = 0.20
    LIFT_NORM_DIVISOR = 3.0


class IntelligentBodyConfig:
    # Weights fusion calibration defaults (used when calibration dataset lacks enough cases)
    DEFAULT_DDC_WEIGHT = 0.30
    DEFAULT_Z_WEIGHT = 0.20
    DEFAULT_ARM_WEIGHT = 0.30
    DEFAULT_BR_WEIGHT = 0.20
    
    # Weight calibration factors
    AUC_OFFSET = 0.45
    MIN_WEIGHT = 0.03
    
    # Risk categorization percentiles (based on historic score distribution)
    RISK_HIGH_PERCENTILE = 90
    RISK_MEDIUM_PERCENTILE = 70
    
    # Explanation and anomaly type thresholds
    TEMPORAL_ANOMALY_Z_THRESHOLD = 0.35
    EXPLANATION_HIGH_Z_THRESHOLD = 0.67

    # Business rule verification thresholds
    BR_SEQ_VIOLATIONS_MAX = 1
    BR_MISSING_STEPS_MAX = 0
    BR_UNUSUAL_RESOURCE_EVENTS_MAX = 0
    BR_TEMP_TOTAL_QUANTILE = 'p90'

    # Sampling and scoring thresholds
    MAX_CALIBRATION_SAMPLES = 2500
    FAST_SCORING_THRESHOLD = 10000




class IterativeBaselineConfig:
    # Iterative baseline split thresholds
    BASELINE_KEEP_FRACTION = 0.80
    BASELINE_THRESHOLD_QUANTILE = 0.95
    
    # Penalties and bonuses for case ranking in iterative baseline selection
    PAYMENT_WEIGHT = 0.25
    COMPLETED_WEIGHT = 0.15
    PENALTY_NO_PAYMENT_WEIGHT = 0.35
    APPEAL_NO_PAYMENT_WEIGHT = 0.20


class ProcessConfig:
    # Profile learning thresholds
    MIN_EDGE_SUPPORT = 0.05
    REQUIRED_ACTIVITY_RATIO = 0.85
    MAX_REPETITION_PERCENTILE = 95
    ALLOWED_RESOURCE_THRESHOLD = 0.01

    # Domain keywords for semantic activity identification
    COMPLETION_KEYWORDS = ['payment', 'archive', 'close']
    APPEAL_KEYWORDS = ['appeal']
    PENALTY_KEYWORDS = ['penalty']
    PAYMENT_KEYWORDS = ['payment']
    
    # RPA resource keywords
    RPA_KEYWORDS = ['bot', 'robot', 'system', 'auto', 'rpa']