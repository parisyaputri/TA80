import numpy as np
import pandas as pd
from pathlib import Path

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
    apply_detection_threshold,
    choose_detection_threshold
)

from utils.baselines import (
    add_lightweight_baselines
)

from utils.result_handler import (
    save_outputs,
    save_evaluation
)

from utils.statistical_tests import (
    run_wilcoxon_test,
    compute_cohens_d,
    apply_bonferroni
)

from configs.paths import (
    OUTPUT_DIR,
    MODEL_DIR,
    RESULT_DIR
)

from utils.evaluation import (
    evaluate_model
)

from utils.evaluation_split import (
    build_case_train_evaluation_split,
    build_threshold_evaluation_split,
    evaluation_scope,
    threshold_scope
)

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

    train_event_mask, _, _ = build_case_train_evaluation_split(
        cleaned_df,
        case_col='_case_id_norm'
    )

    train_events = cleaned_df.loc[
        train_event_mask
    ].copy()

    train_case_ids = set(
        train_events['_case_id_norm']
        .astype(str)
        .unique()
    )

    # =====================================================
    # PROFILE LEARNING
    # =====================================================

    profile_df = train_events

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
    train_feature_df = feature_df[
        feature_df['case_id']
        .astype(str)
        .isin(train_case_ids)
    ].copy()

    train_feature_for_learning = train_feature_df.drop(
        columns=['label'],
        errors='ignore'
    )

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
        train_feature_for_learning,
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
        train_feature_for_learning
    )

    ib = IntelligentBody(
        dt,
        ddc,
        mv_arm
    )

    ib.calibrate_weights(
        train_feature_for_learning,
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

    calibration_mask, test_mask, has_holdout_test = (
        build_threshold_evaluation_split(
            final_df
        )
    )

    threshold_df = final_df.loc[
        calibration_mask
    ].drop(
        columns=['label'],
        errors='ignore'
    )

    threshold, threshold_method = (
        choose_detection_threshold(
            threshold_df
        )
    )

    final_df['threshold'] = threshold

    final_df['threshold_method'] = (
        threshold_method
    )

    current_threshold_scope = threshold_scope(
        has_holdout_test
    )

    final_df['threshold_scope'] = current_threshold_scope

    final_df['predicted_label'] = apply_detection_threshold(
        final_df,
        'anomaly_score',
        threshold,
        threshold_method
    )

    if 'risk_level' not in final_df.columns:
        final_df['risk_level'] = 'Low'

    # =====================================================
    # BASELINES
    # =====================================================

    final_df = add_lightweight_baselines(
        final_df,
        mv_arm,
        calibration_mask=calibration_mask
    )

    # =====================================================
    # EVALUATION
    # =====================================================

    metrics = None
    wilcoxon_stat = None
    p_value = None
    cohens_d = None
    corrected_p = None

    evaluation_df = final_df.loc[
        test_mask
    ]

    if 'label' in final_df.columns:

        metrics = evaluate_model(
            y_true=evaluation_df['label'],
            y_pred=evaluation_df['predicted_label'],
            y_scores=evaluation_df['anomaly_score']
        )
        
        # =====================================================
        # STATISTICAL TESTING
        # =====================================================

        if (
            metrics is not None and
            'single_arm_score' in final_df.columns
        ):

            baseline_scores = (
                evaluation_df['single_arm_score']
            )

            proposed_scores = (
                evaluation_df['anomaly_score']
            )

            wilcoxon_stat, p_value = (
                run_wilcoxon_test(
                    proposed_scores,
                    baseline_scores
                )
            )

            cohens_d = compute_cohens_d(
                proposed_scores,
                baseline_scores
            )

            corrected_p = apply_bonferroni(
                [p_value]
            )

            print(
                "\n===== STATISTICAL TESTING ====="
            )

            print(
                f"Wilcoxon Statistic : "
                f"{wilcoxon_stat:.4f}"
            )

            print(
                f"P-Value            : "
                f"{p_value:.6f}"
            )

            print(
                f"Cohen's d          : "
                f"{cohens_d:.4f}"
            )

            print(
                f"Bonferroni Correct : "
                f"{corrected_p[0]:.6f}"
            )
    # =====================================================
    # SAVE EVALUATION REPORT
    # =====================================================

    if 'label' in final_df.columns:

        dataset_name = Path(
            csv_path
        ).stem

        evaluation_path = (
            RESULT_DIR /
            f"{dataset_name}_eval.txt"
        )

        current_evaluation_scope = (
            evaluation_scope(
                has_holdout_test
            )
        )

        evaluations = [
            {
                'title': f'DT-IB ADAPTIVE MODEL ({current_evaluation_scope})',
                'y_true': evaluation_df['label'],
                'y_pred': evaluation_df['predicted_label'],
                'y_scores': evaluation_df['anomaly_score'],
                'statistical_testing': {
                    'wilcoxon': wilcoxon_stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'bonferroni': corrected_p[0]
                }
            },
            {
                'title': f'BASELINE: STATIC DC ({current_evaluation_scope})',
                'y_true': evaluation_df['label'],
                'y_pred': evaluation_df['static_dc_predicted_label'],
                'y_scores': evaluation_df['static_dc_score']
            },
            {
                'title': f'BASELINE: SINGLE-VIEW ARM ({current_evaluation_scope})',
                'y_true': evaluation_df['label'],
                'y_pred': evaluation_df['single_arm_predicted_label'],
                'y_scores': evaluation_df['single_arm_score']
            }
        ]

        save_evaluation(
            evaluations,
            evaluation_path
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
        'threshold': threshold,
        'threshold_method': threshold_method,
        'threshold_scope': current_threshold_scope,
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
        'threshold_scope': current_threshold_scope,
        'calibration_rows': int(calibration_mask.sum()),
        'evaluation_rows': int(test_mask.sum()),
        'result_file': str(output_path.name)
    }
