import numpy as np
import pandas as pd

from utils.thresholding import (
    apply_detection_threshold,
    choose_detection_threshold
)

from configs.model_config import DDCConfig, IntelligentBodyConfig
from models.lstm_baseline import LSTMBaseline
from models.transformer_baseline import TransformerBaseline

def add_lightweight_baselines(final_df, mv_arm, calibration_mask=None):
    static_components = pd.DataFrame(index=final_df.index)
    static_components['cf'] = (
        (final_df['cf_seq_violations'] > 0).astype(float)
        + (final_df['cf_missing_steps'] > 0).astype(float)
        + (final_df['cf_duplicate_steps'] > 0).astype(float)
    ) / DDCConfig.SEVERITY_Z_DIVISOR
    static_components['temporal'] = (
        final_df['temp_total_hrs']
        > final_df['temp_total_hrs'].quantile(IntelligentBodyConfig.RISK_HIGH_PERCENTILE / 100.0)
    ).astype(float)
    static_components['resource'] = (
        final_df['res_unusual_activity_count'] > 0
    ).astype(float)

    final_df['static_dc_score'] = static_components.mean(axis=1).round(4)

    single_arm = mv_arm.score_dataframe(final_df, single_view=True)

    final_df['single_arm_score'] = single_arm['arm_score'].round(4)
    final_df['single_arm_rules_hit'] = single_arm['arm_rules_hit']

    threshold_df = final_df

    if calibration_mask is not None:
        threshold_df = final_df.loc[calibration_mask]

    threshold_features = threshold_df.drop(
        columns=['label'],
        errors='ignore'
    )

    static_threshold, static_threshold_method = choose_detection_threshold(
        threshold_features,
        'static_dc_score'
    )

    positive_single_scores = final_df.loc[
        final_df['single_arm_score'] > 0,
        'single_arm_score'
    ]

    if (
        'single_arm_score' in threshold_features.columns
    ):
        arm_threshold, arm_threshold_method = choose_detection_threshold(
            threshold_features,
            'single_arm_score'
        )
    else:
        arm_threshold = (
            round(float(positive_single_scores.min()), 4)
            if not positive_single_scores.empty
            else 1.0
        )
        arm_threshold_method = 'single_arm_score_min_positive'

    final_df['static_dc_predicted_label'] = apply_detection_threshold(
        final_df,
        'static_dc_score',
        static_threshold,
        static_threshold_method
    )
    final_df['single_arm_predicted_label'] = apply_detection_threshold(
        final_df,
        'single_arm_score',
        arm_threshold,
        arm_threshold_method
    )

    final_df['static_dc_threshold'] = static_threshold
    final_df['static_dc_threshold_method'] = static_threshold_method
    final_df['single_arm_threshold'] = arm_threshold
    final_df['single_arm_threshold_method'] = arm_threshold_method

    return final_df

def run_lstm_baseline(X_train, X_test):

    model = LSTMBaseline()

    model.fit(X_train)

    return model.predict(X_test)

def run_transformer_baseline(
    X_train,
    y_train,
    X_test
):

    model = TransformerBaseline()

    model.fit(X_train, y_train)

    return model.predict(X_test)
