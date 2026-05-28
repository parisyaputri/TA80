class DDCConfig:
    # Thresholds for statistical outliers
    Z_THRESHOLD = 2.5                    # Z-score of 2.5 corresponds to the 98.7% confidence interval, defining clear statistical outliers
    IQR_MULTIPLIER = 2.0                 # 2.0 IQR represents moderate-to-extreme outliers (between mild 1.5 IQR and extreme 3.0 IQR)
    EWMA_MARGIN = 0.10                   # 10% margin allowance around EWMA predictions to accommodate standard temporal variance without false alarms
    
    # Severity calculation normalization factors
    SEVERITY_Z_DIVISOR = 3.0             # Normalizes Z-score deviations (a Z-score of 3.0 represents a 99.7% extreme deviation, mapping it to 1.0)
    SEVERITY_VIOL_DIVISOR = 4.0          # Normalizes the count of rule violations, assuming 4 or more violations represents maximum severity (1.0)

    # Hard rules bounds
    ALLOWED_MISSING_STEPS_MAX = 0        # Normal traces must not skip any required activities (strict control-flow compliance)
    ALLOWED_DUPLICATE_STEPS_MAX = 2      # Allows up to 2 duplicate activities to tolerate standard retry mechanisms or admin corrections
    ALLOWED_SEQ_VIOLATIONS_MAX = 1       # Allows at most 1 sequence violation for mild, acceptable variations in the execution path
    ALLOWED_UNUSUAL_RESOURCE_EVENTS_MAX = 0 # Strictly 0 unusual resource events allowed to enforce standard authorization policies and segregation of duties

    # EWMA settings
    EWMA_ALPHA = 0.20                    # Smoothing factor of 0.20 gives a moderate memory of past events, prioritizing recent trends (standard range: 0.05-0.30)



class ARMConfig:
    # Association Rule Mining parameters
    MIN_SUPPORT = 0.06                   # 6% minimum support filters out rare activities/noise while preserving standard process constraint pairs
    MIN_CONFIDENCE = 0.65                # 65% minimum confidence ensures mined rules represent strong correlations without over-constraining the model
    MIN_LIFT = 1.05                      # 1.05 minimum lift ensures rules reflect positive correlations, discarding trivial independent rules (lift <= 1.0)
    MAX_RULES = 40                       # Caps the rule set to the top 40 rules to optimize constraint-checking runtime and avoid redundant rules
    
    # Severity weighting for ARM rule hits
    CONFIDENCE_WEIGHT = 0.50             # 50% weight on confidence as it directly measures the correctness/accuracy of the mined association rule
    LIFT_WEIGHT = 0.30                   # 30% weight on lift as it measures the relative strength of correlation between activities
    SUPPORT_WEIGHT = 0.20                # 20% weight on support as it represents the popularity/frequency of the pattern in the log
    LIFT_NORM_DIVISOR = 3.0              # Divisor of 3.0 normalizes lift values (commonly ranging from 1 to 3) to a [0, 1] scale



class IntelligentBodyConfig:
    # Weights fusion calibration defaults (used when calibration dataset lacks enough cases)
    DEFAULT_DDC_WEIGHT = 0.30            # Default 30% weight on DDC (strict rule violations are a strong indicator of anomalies)
    DEFAULT_Z_WEIGHT = 0.20              # Default 20% weight on Z-score (sensitive to temporal delay but contains more natural variance/noise)
    DEFAULT_ARM_WEIGHT = 0.30            # Default 30% weight on Association Rules (highly effective for trace co-occurrence violations)
    DEFAULT_BR_WEIGHT = 0.20             # Default 20% weight on Business Rules (balanced checking of domain constraints)
    
    # Weight calibration factors
    AUC_OFFSET = 0.45                    # Subtracted from AUC values to represent random guessing baseline (AUC = 0.50) with a 5% safety margin
    MIN_WEIGHT = 0.03                    # Minimum 3% weight for any model component to ensure no view is completely excluded from the fusion process
    
    # Risk categorization percentiles (based on historic score distribution)
    RISK_HIGH_PERCENTILE = 90            # Top 10% highest scores are mapped to High Risk (matching standard anomaly contamination rates of 5-10%)
    RISK_MEDIUM_PERCENTILE = 70          # Scores between 70% and 90% are mapped to Medium Risk, capturing borderline cases
    
    # Explanation and anomaly type thresholds
    TEMPORAL_ANOMALY_Z_THRESHOLD = 0.35  # Individual activity delays with a severity score > 0.35 are classified as Temporal anomalies
    EXPLANATION_HIGH_Z_THRESHOLD = 0.67  # 0.67 corresponds to the 75th percentile of the standard normal curve, flagging highly significant deviations

    # Business rule verification thresholds
    BR_SEQ_VIOLATIONS_MAX = 1            # Domain business rule: max 1 sequence violation allowed before flagging
    BR_MISSING_STEPS_MAX = 0             # Domain business rule: 0 missing steps allowed
    BR_UNUSUAL_RESOURCE_EVENTS_MAX = 0   # Domain business rule: 0 unauthorized resource activities allowed
    BR_TEMP_TOTAL_QUANTILE = 'p90'       # Domain business rule: uses the 90th percentile of case durations as the maximum SLA limit

    # Sampling and scoring thresholds
    MAX_CALIBRATION_SAMPLES = 2500       # Caps calibration sample size to 2500 to keep the weight solver fast during web UI uploads
    FAST_SCORING_THRESHOLD = 10000       # Above 10,000 cases, switches to matrix vectorization to prevent web UI browser timeouts




class IterativeBaselineConfig:
    # Iterative baseline split thresholds
    BASELINE_KEEP_FRACTION = 0.80        # Keeps the 80% most normal-looking cases in Round 1 to build a clean baseline profile in Round 2
    BASELINE_THRESHOLD_QUANTILE = 0.95   # Sets the anomaly detection threshold to flag the top 5% most anomalous cases under the null hypothesis
    
    # Penalties and bonuses for case ranking in iterative baseline selection
    PAYMENT_WEIGHT = 0.25                # Having a payment reduces the anomaly score by 25% (strong indication of standard process completion)
    COMPLETED_WEIGHT = 0.15              # Trace reaching its standard closing activity reduces the anomaly score by 15%
    PENALTY_NO_PAYMENT_WEIGHT = 0.35     # Penalties issued without matching payments increase the anomaly score by 35%
    APPEAL_NO_PAYMENT_WEIGHT = 0.20      # Appeals created without matching payments increase the anomaly score by 20%


class ProcessConfig:
    # Profile learning thresholds
    MIN_EDGE_SUPPORT = 0.05              # Edges present in < 5% of cases are ignored to prevent infrequent paths from polluting standard profiles
    REQUIRED_ACTIVITY_RATIO = 0.85       # Activities present in > 85% of traces are classified as required steps for process execution
    MAX_REPETITION_PERCENTILE = 95       # Binds maximum allowed repetitions of an activity to its 95th percentile in baseline data
    ALLOWED_RESOURCE_THRESHOLD = 0.01    # Resources processing > 1% of an activity type are registered as authorized for that activity

    # Domain keywords for semantic activity identification
    COMPLETION_KEYWORDS = ['payment', 'archive', 'close'] # Standard words identifying standard trace closure
    APPEAL_KEYWORDS = ['appeal']         # Identifies appeal steps in judicial/billing processes
    PENALTY_KEYWORDS = ['penalty']       # Identifies penalty or fine-issuance activities
    PAYMENT_KEYWORDS = ['payment']       # Identifies payment or billing completion activities
    
    # RPA resource keywords
    RPA_KEYWORDS = ['bot', 'robot', 'system', 'auto', 'rpa'] # Keywords to identify automated bot executions vs human actions