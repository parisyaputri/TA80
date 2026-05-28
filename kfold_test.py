import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from configs.model_config import IntelligentBodyConfig
from models.ddc import DynamicDeclarativeConstraints
from models.digital_twin import DigitalTwin
from models.intelligent_body import IntelligentBody
from models.mv_arm import MVARMiner
from utils.baselines import add_lightweight_baselines
from models.lstm_baseline import LSTMBaseline
from models.transformer_baseline import TransformerBaseline
from utils.feature_engineering import (
    build_case_features,
    learn_control_flow_profile,
    learn_resource_profile,
    replay_event_states,
)
from utils.iterative_baseline import (
    apply_process_completion_rules,
    choose_iterative_baseline_threshold,
    select_stable_baseline_cases,
)
from utils.preprocessing import detect_event_log_columns, preprocess_event_log
from utils.thresholding import apply_detection_threshold, choose_detection_threshold


NUMERIC_COLS = [
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
    'expense',
]


def _case_labels(cleaned_df, label_col):
    if label_col is None:
        raise ValueError('Dataset tidak punya kolom label.')

    labels = (
        cleaned_df
        .groupby('_case_id_norm', sort=False)[label_col]
        .apply(lambda values: 'deviant'
               if values.astype(str).str.lower().eq('deviant').any()
               else 'regular')
        .reset_index(name='label')
    )

    labels = labels[labels['label'].isin(['regular', 'deviant'])]

    if labels['label'].nunique() != 2:
        raise ValueError(
            'K-fold butuh dua kelas label: regular dan deviant.'
        )

    return labels


def _training_profile(cleaned_df, label_col):
    return cleaned_df


def _build_fold_features(train_events, test_events, label_col, profile_events=None):
    profile_df = profile_events

    if profile_df is None:
        profile_df = _training_profile(train_events, label_col)

    cf_profile = learn_control_flow_profile(
        profile_df.groupby('_case_id_norm', sort=False)
    )
    resource_profile = learn_resource_profile(profile_df)

    train_states = replay_event_states(train_events)
    test_states = replay_event_states(test_events)

    train_features = build_case_features(
        train_events,
        cf_profile,
        resource_profile,
        train_states,
        label_col,
    )
    test_features = build_case_features(
        test_events,
        cf_profile,
        resource_profile,
        test_states,
        label_col,
    )

    return train_features, test_features, cf_profile, resource_profile, train_states


def _fit_fold_model(
    train_features,
    cf_profile,
    resource_profile,
    train_states,
    baseline_case_ids=None
):
    learning_features = train_features

    if baseline_case_ids is not None:
        learning_features = train_features[
            train_features['case_id']
            .astype(str)
            .isin(baseline_case_ids)
        ]

    model_train_df = learning_features.drop(
        columns=['label'],
        errors='ignore',
    )

    dt = DigitalTwin()
    dt.fit(model_train_df, NUMERIC_COLS)
    dt.seed_states(train_states)

    ddc = DynamicDeclarativeConstraints()
    ddc.fit(
        dt,
        NUMERIC_COLS,
        train_df=learning_features,
        cf_profile=cf_profile,
        resource_profile=resource_profile,
    )

    mv_arm = MVARMiner()
    mv_arm.fit(model_train_df)

    ib = IntelligentBody(dt, ddc, mv_arm)
    ib.calibrate_weights(train_features, NUMERIC_COLS)

    return ib, mv_arm


def _score_features(features, ib, mv_arm):
    scoring_df = features.copy()
    precomputed_arm = mv_arm.score_dataframe(scoring_df)

    scoring_df['_pre_arm_score'] = precomputed_arm['arm_score']
    scoring_df['_pre_arm_rules_hit'] = precomputed_arm['arm_rules_hit']
    scoring_df['_pre_violated_arm_rules'] = precomputed_arm['violated_arm_rules']

    if len(scoring_df) > IntelligentBodyConfig.FAST_SCORING_THRESHOLD:
        result_df = ib.score_all_fast(scoring_df)
    else:
        result_df = ib.score_all(scoring_df)

    return pd.merge(
        features,
        result_df,
        on='case_id',
    )


def _classification_metrics(y_true, y_pred, y_scores):
    y_true_binary = y_true.eq('deviant').astype(int)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(
            y_true,
            y_pred,
            pos_label='deviant',
            zero_division=0,
        ),
        'recall': recall_score(
            y_true,
            y_pred,
            pos_label='deviant',
            zero_division=0,
        ),
        'f1': f1_score(
            y_true,
            y_pred,
            pos_label='deviant',
            zero_division=0,
        ),
        'mcc': matthews_corrcoef(y_true, y_pred),
    }

    try:
        metrics['auc_roc'] = roc_auc_score(y_true_binary, y_scores)
    except ValueError:
        metrics['auc_roc'] = np.nan

    try:
        metrics['auc_pr'] = average_precision_score(y_true_binary, y_scores)
    except ValueError:
        metrics['auc_pr'] = np.nan

    return metrics


def run_kfold(csv_path, n_splits=5, random_state=42, output_dir='outputs/kfold'):
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        csv_path,
        sep=None,
        engine='python',
    )

    columns = detect_event_log_columns(df)
    label_col = columns['label']
    cleaned_df = preprocess_event_log(df, columns)
    labels = _case_labels(cleaned_df, label_col)

    class_counts = labels['label'].value_counts()
    min_class_count = int(class_counts.min())

    if min_class_count < n_splits:
        raise ValueError(
            f'Jumlah kelas terkecil hanya {min_class_count}; '
            f'tidak cukup untuk {n_splits}-fold.'
        )

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    rows = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(labels['_case_id_norm'], labels['label']),
        start=1,
    ):
        train_cases = set(labels.iloc[train_idx]['_case_id_norm'])
        test_cases = set(labels.iloc[test_idx]['_case_id_norm'])

        train_events = cleaned_df[
            cleaned_df['_case_id_norm'].isin(train_cases)
        ].copy()
        test_events = cleaned_df[
            cleaned_df['_case_id_norm'].isin(test_cases)
        ].copy()

        (
            initial_train_features,
            _,
            initial_cf_profile,
            initial_resource_profile,
            initial_train_states,
        ) = _build_fold_features(
            train_events,
            test_events,
            label_col,
        )

        ib, mv_arm = _fit_fold_model(
            initial_train_features,
            initial_cf_profile,
            initial_resource_profile,
            initial_train_states,
        )

        initial_train_scored = _score_features(
            initial_train_features,
            ib,
            mv_arm,
        )

        stable_case_ids = select_stable_baseline_cases(
            initial_train_scored
        )

        stable_train_events = train_events[
            train_events['_case_id_norm']
            .astype(str)
            .isin(stable_case_ids)
        ].copy()

        (
            train_features,
            test_features,
            cf_profile,
            resource_profile,
            train_states,
        ) = _build_fold_features(
            train_events,
            test_events,
            label_col,
            profile_events=stable_train_events,
        )

        ib, mv_arm = _fit_fold_model(
            train_features,
            cf_profile,
            resource_profile,
            train_states,
            baseline_case_ids=stable_case_ids,
        )

        train_scored = _score_features(train_features, ib, mv_arm)
        test_scored = _score_features(test_features, ib, mv_arm)

        threshold, threshold_method = choose_iterative_baseline_threshold(
            train_scored,
            stable_case_ids,
        )

        test_scored['threshold'] = threshold
        test_scored['threshold_method'] = threshold_method
        test_scored['predicted_label'] = apply_detection_threshold(
            test_scored,
            'anomaly_score',
            threshold,
            threshold_method,
        )
        test_scored['predicted_label'] = apply_process_completion_rules(
            test_scored
        )

        test_scored = add_lightweight_baselines(
            test_scored,
            mv_arm,
            calibration_mask=None,
        )

        # Train and predict LSTM Baseline
        stable_train_features = train_features[
            train_features['case_id'].astype(str).isin(stable_case_ids)
        ]
        if len(stable_train_features) == 0:
            stable_train_features = train_features
            
        lstm = LSTMBaseline()
        lstm.fit(stable_train_features[NUMERIC_COLS])
        test_scored['lstm_score'] = lstm.predict_score(test_scored[NUMERIC_COLS])
        lstm_preds = lstm.predict(test_scored[NUMERIC_COLS])
        test_scored['lstm_predicted_label'] = ['deviant' if p == 1 else 'regular' for p in lstm_preds]

        # Train and predict Transformer Baseline
        transformer = TransformerBaseline()
        transformer.fit(train_features[NUMERIC_COLS], train_features['label'])
        test_scored['transformer_score'] = transformer.predict_score(test_scored[NUMERIC_COLS])
        transformer_preds = transformer.predict(test_scored[NUMERIC_COLS])
        test_scored['transformer_predicted_label'] = ['deviant' if p == 1 else 'regular' for p in transformer_preds]

        # Calculate baseline metrics
        metrics = _classification_metrics(
            test_scored['label'],
            test_scored['predicted_label'],
            test_scored['anomaly_score'],
        )
        lstm_metrics = _classification_metrics(
            test_scored['label'],
            test_scored['lstm_predicted_label'],
            test_scored['lstm_score']
        )
        transformer_metrics = _classification_metrics(
            test_scored['label'],
            test_scored['transformer_predicted_label'],
            test_scored['transformer_score']
        )

        row = {
            'fold': fold,
            'train_cases': len(train_features),
            'test_cases': len(test_features),
            'train_regular': int(train_features['label'].eq('regular').sum()),
            'train_deviant': int(train_features['label'].eq('deviant').sum()),
            'test_regular': int(test_features['label'].eq('regular').sum()),
            'test_deviant': int(test_features['label'].eq('deviant').sum()),
            'stable_baseline_cases': len(stable_case_ids),
            'threshold': threshold,
            'threshold_method': threshold_method,
            **metrics,
            **{f'lstm_{k}': v for k, v in lstm_metrics.items()},
            **{f'transformer_{k}': v for k, v in transformer_metrics.items()}
        }
        rows.append(row)


        print(
            f"Fold {fold}: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"f1={metrics['f1']:.4f}, "
            f"threshold={threshold:.4f}"
        )

    results = pd.DataFrame(rows)

    summary = pd.DataFrame([{
        'dataset': csv_path.name,
        'n_splits': n_splits,
        'random_state': random_state,
        'total_cases': len(labels),
        'regular_cases': int(class_counts.get('regular', 0)),
        'deviant_cases': int(class_counts.get('deviant', 0)),
        'mean_accuracy': results['accuracy'].mean(),
        'std_accuracy': results['accuracy'].std(ddof=1),
        'min_accuracy': results['accuracy'].min(),
        'max_accuracy': results['accuracy'].max(),
        'best_fold': int(results.loc[results['accuracy'].idxmax(), 'fold']),
        'mean_f1': results['f1'].mean(),
        'mean_auc_roc': results['auc_roc'].mean(),
        'mean_auc_pr': results['auc_pr'].mean(),
        'mean_lstm_accuracy': results['lstm_accuracy'].mean(),
        'mean_lstm_f1': results['lstm_f1'].mean(),
        'mean_lstm_auc_roc': results['lstm_auc_roc'].mean(),
        'mean_lstm_auc_pr': results['lstm_auc_pr'].mean(),
        'mean_transformer_accuracy': results['transformer_accuracy'].mean(),
        'mean_transformer_f1': results['transformer_f1'].mean(),
        'mean_transformer_auc_roc': results['transformer_auc_roc'].mean(),
        'mean_transformer_auc_pr': results['transformer_auc_pr'].mean(),
        'mean_stable_baseline_cases': results['stable_baseline_cases'].mean(),
    }])

    dataset_name = csv_path.stem
    results_path = output_dir / f'{dataset_name}_kfold5_results.csv'
    summary_path = output_dir / f'{dataset_name}_kfold5_summary.csv'

    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)

    print('\n===== 5-FOLD SUMMARY =====')
    print(summary.to_string(index=False))
    print(f'\nDetail per fold disimpan di: {results_path}')
    print(f'Summary disimpan di       : {summary_path}')

    return results, summary


def parse_args():
    parser = argparse.ArgumentParser(
        description='Uji kestabilan model dengan stratified 5-fold CV.'
    )
    parser.add_argument(
        'csv_path',
        nargs='?',
        default='data/traffic_fines_1.csv',
        help='Path CSV dataset. Default: data/traffic_fines_1.csv',
    )
    parser.add_argument(
        '--folds',
        type=int,
        default=5,
        help='Jumlah fold. Default: 5',
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Seed shuffle StratifiedKFold. Default: 42',
    )
    parser.add_argument(
        '--output-dir',
        default='outputs/kfold',
        help='Folder output CSV hasil k-fold. Default: outputs/kfold',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_kfold(
        args.csv_path,
        n_splits=args.folds,
        random_state=args.random_state,
        output_dir=args.output_dir,
    )
