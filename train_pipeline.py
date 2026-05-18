from pathlib import Path

import numpy as np
import pandas as pd

from models.digital_twin import DigitalTwin
from models.ddc import DynamicDeclarativeConstraints
from models.mv_arm import MVARMiner
from models.intelligent_body import IntelligentBody

from utils.preprocessing import (
    detect_event_log_columns,
    preprocess_event_log
)

from utils.feature_engineering import (
    learn_control_flow_profile,
    learn_resource_profile,
    replay_event_states,
    build_case_features
)

from utils.thresholding import (
    choose_detection_threshold
)

from utils.baselines import (
    add_lightweight_baselines
)

from utils.save_results import (
    save_outputs
)


BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / 'outputs'

MODEL_DIR = OUTPUT_DIR / 'model'
RESULT_DIR = OUTPUT_DIR / 'results'

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def train_and_detect(csv_path):

    # =====================================================
    # LOAD DATA
    # =====================================================

    df = pd.read_csv(
        csv_path,
        sep=None,
        engine='python'
    )

    columns = detect_event_log_columns(df)

    label_col = columns['label']

    cleaned_df = preprocess_event_log(
        df,
        columns
    )

    # =====================================================
    # PROFILE LEARNING
    # =====================================================

    profile_df = cleaned_df

    if label_col is not None:

        labels_for_profile = (
            cleaned_df[label_col]
            .astype(str)
            .str.lower()
        )

        regular_df = cleaned_df[
            labels_for_profile == 'regular'
        ]

        if not regular_df.empty:
            profile_df = regular_df

    profile_grouped = profile_df.groupby(
        '_case_id_norm',
        sort=False
    )

    cf_profile = learn_control_flow_profile(
        profile_grouped
    )

    resource_profile = learn_resource_profile(
        profile_df
    )

    case_states = replay_event_states(
        cleaned_df
    )

    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================

    feature_df = build_case_features(
        cleaned_df,
        cf_profile,
        resource_profile,
        case_states,
        label_col
    )

    train_feature_df = feature_df

    if 'label' in feature_df.columns:

        regular_feature_df = feature_df[
            feature_df['label'] == 'regular'
        ]

        if not regular_feature_df.empty:
            train_feature_df = regular_feature_df

    # =====================================================
    # NUMERIC FEATURES
    # =====================================================

    numeric_cols = [
        'cf_n_events',
        'cf_seq_violations',
        'cf_missing_steps',
        'cf_duplicate_steps',
        'cf_wrong_order_ratio',
        'temp_total_hrs',
        'temp_max_step_hrs',
        'temp_std_step_hrs',
        'res_n_resources',
        'res_single_resource',
        'res_many_resources',
        'res_dominant_resource_ratio',
        'res_unusual_activity_count',
        'res_unusual_activity_ratio',
        'res_workload_share',
        'res_resource_rarity',
        'amount',
        'expense'
    ]

    # =====================================================
    # MODEL TRAINING
    # =====================================================

    dt = DigitalTwin()

    dt.fit(
        train_feature_df,
        numeric_cols
    )

    dt.seed_states(case_states)

    ddc = DynamicDeclarativeConstraints()

    ddc.fit(
        dt,
        numeric_cols,
        cf_profile=cf_profile,
        resource_profile=resource_profile
    )

    mv_arm = MVARMiner()

    mv_arm.fit(
        train_feature_df
    )

    ib = IntelligentBody(
        dt,
        ddc,
        mv_arm
    )

    ib.calibrate_weights(
        feature_df,
        numeric_cols
    )

    # =====================================================
    # PRECOMPUTE ARM
    # =====================================================

    precomputed_arm = mv_arm.score_dataframe(
        feature_df
    )

    scoring_df = feature_df.copy()

    scoring_df['_pre_arm_score'] = (
        precomputed_arm['arm_score']
    )

    scoring_df['_pre_arm_rules_hit'] = (
        precomputed_arm['arm_rules_hit']
    )

    scoring_df['_pre_violated_arm_rules'] = (
        precomputed_arm['violated_arm_rules']
    )

    # =====================================================
    # ANOMALY SCORING
    # =====================================================

    if len(scoring_df) > 10000:

        result_df = ib.score_all_fast(
            scoring_df
        )

    else:

        result_df = ib.score_all(
            scoring_df
        )

    # =====================================================
    # MERGE RESULTS
    # =====================================================

    final_df = pd.merge(
        feature_df,
        result_df,
        on='case_id'
    )

    # =====================================================
    # THRESHOLDING
    # =====================================================

    threshold, threshold_method = (
        choose_detection_threshold(
            final_df
        )
    )

    final_df['threshold'] = threshold

    final_df['threshold_method'] = (
        threshold_method
    )

    final_df['predicted_label'] = np.where(
        final_df['anomaly_score'] >= threshold,
        'deviant',
        'regular'
    )

    if 'risk_level' not in final_df.columns:
        final_df['risk_level'] = 'Low'

    # =====================================================
    # BASELINES
    # =====================================================

    final_df = add_lightweight_baselines(
        final_df,
        mv_arm
    )

    # =====================================================
    # SAVE OUTPUTS
    # =====================================================

    model_bundle = {
        'digital_twin': dt,
        'ddc': ddc,
        'mv_arm': mv_arm,
        'intelligent_body': ib,
        'control_flow_profile': cf_profile,
        'resource_profile': resource_profile,
        'numeric_cols': numeric_cols,
    }

    output_path = save_outputs(
        model_bundle,
        final_df,
        MODEL_DIR,
        RESULT_DIR
    )

    # =====================================================
    # SUMMARY
    # =====================================================

    anomaly_count = int(
        final_df['predicted_label']
        .eq('deviant')
        .sum()
    )

    normal_count = (
        len(final_df)
        - anomaly_count
    )

    return {
        'total_rows': len(final_df),
        'anomaly_count': anomaly_count,
        'normal_count': normal_count,
        'threshold': threshold,
        'threshold_method': threshold_method,
        'result_file': str(output_path.name)
    }