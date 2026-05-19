import numpy as np
import pandas as pd

from utils.thresholding import (
    choose_detection_threshold
)

from models.lstm_baseline import LSTMBaseline
from models.transformer_baseline import TransformerBaseline

def add_lightweight_baselines(final_df, mv_arm):
    static_components = pd.DataFrame(index=final_df.index)
    static_components['cf'] = (
        (final_df['cf_seq_violations'] > 0).astype(float)
        + (final_df['cf_missing_steps'] > 0).astype(float)
        + (final_df['cf_duplicate_steps'] > 0).astype(float)
    ) / 3.0
    static_components['temporal'] = (
        final_df['temp_total_hrs']
        > final_df['temp_total_hrs'].quantile(0.90)
    ).astype(float)
    static_components['resource'] = (
        final_df['res_unusual_activity_count'] > 0
    ).astype(float)

    final_df['static_dc_score'] = static_components.mean(axis=1).round(4)

    single_arm = mv_arm.score_dataframe(final_df, single_view=True)

    final_df['single_arm_score'] = single_arm['arm_score'].round(4)
    final_df['single_arm_rules_hit'] = single_arm['arm_rules_hit']

    static_threshold, _ = choose_detection_threshold(final_df, 'static_dc_score')
    positive_single_scores = final_df.loc[
        final_df['single_arm_score'] > 0,
        'single_arm_score'
    ]
    arm_threshold = (
        round(float(positive_single_scores.min()), 4)
        if not positive_single_scores.empty
        else 1.0
    )

    final_df['static_dc_predicted_label'] = np.where(
        final_df['static_dc_score'] >= static_threshold,
        'deviant',
        'regular'
    )
    final_df['single_arm_predicted_label'] = np.where(
        final_df['single_arm_score'] >= arm_threshold,
        'deviant',
        'regular'
    )

    final_df['static_dc_threshold'] = static_threshold
    final_df['single_arm_threshold'] = arm_threshold

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