import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from tf_models import (
    DigitalTwin,
    DynamicDeclarativeConstraints,
    MVARMiner,
    IntelligentBody
)


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / 'dataOutput' / 'model'
RESULT_DIR = BASE_DIR / 'dataOutput' / 'results'

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)


def safe_resource_value(value):
    if pd.isna(value):
        return 'unknown'

    text = str(value).strip().lower()
    return text if text else 'unknown'


def _normalise_name(value):
    if pd.isna(value):
        return 'unknown'

    return str(value).strip().lower()


def detect_event_log_columns(df):
    case_col = None
    activity_col = None
    timestamp_col = None
    resource_col = None
    label_col = None

    for col in df.columns:
        col_lower = col.lower()

        if label_col is None and 'label' in col_lower:
            label_col = col

        if (
            case_col is None
            and 'case' in col_lower
            and (
                'id' in col_lower
                or 'concept:name' in col_lower
            )
        ):
            case_col = col

        if (
            activity_col is None
            and (
                col_lower in ['activity', 'concept:name']
                or 'activity' in col_lower
            )
        ):
            activity_col = col

        if (
            resource_col is None
            and any(keyword in col_lower for keyword in [
                'resource',
                'org:resource',
                'user',
                'employee',
                'staff',
                'worker',
                'officer',
                'agent',
                'actor'
            ])
        ):
            resource_col = col

    timestamp_candidates = []

    for col in df.columns:
        col_lower = col.lower()

        if not any(keyword in col_lower for keyword in [
            'timestamp',
            'complete timestamp',
            'time:timestamp',
            'date'
        ]):
            continue

        parsed = pd.to_datetime(df[col], errors='coerce')
        parse_ratio = parsed.notna().mean()

        if parse_ratio == 0:
            continue

        name_score = 0

        if 'complete timestamp' in col_lower:
            name_score += 4

        if 'timestamp' in col_lower:
            name_score += 3

        if 'date' in col_lower:
            name_score += 1

        if pd.api.types.is_numeric_dtype(df[col]):
            name_score -= 5

        timestamp_candidates.append((name_score + parse_ratio, col))

    if timestamp_candidates:
        timestamp_col = max(timestamp_candidates)[1]

    if case_col is None:
        case_col = df.columns[0]

    if activity_col is None:
        activity_col = df.columns[1]

    return {
        'case': case_col,
        'activity': activity_col,
        'timestamp': timestamp_col,
        'resource': resource_col,
        'label': label_col,
    }


def preprocess_event_log(df, columns):
    case_col = columns['case']
    activity_col = columns['activity']
    timestamp_col = columns['timestamp']
    resource_col = columns['resource']

    cleaned = df.copy()
    cleaned = cleaned.dropna(subset=[case_col, activity_col])
    cleaned = cleaned.drop_duplicates()

    if timestamp_col is not None:
        cleaned['_parsed_timestamp'] = pd.to_datetime(
            cleaned[timestamp_col],
            errors='coerce'
        )
        cleaned = cleaned.dropna(subset=['_parsed_timestamp'])
        cleaned = cleaned.sort_values([case_col, '_parsed_timestamp'])
    else:
        cleaned['_parsed_timestamp'] = pd.NaT
        cleaned = cleaned.sort_values([case_col])

    cleaned['_case_id_norm'] = cleaned[case_col].astype(str)
    cleaned['_activity_norm'] = cleaned[activity_col].apply(_normalise_name)

    if resource_col is not None:
        cleaned['_resource_norm'] = cleaned[resource_col].apply(safe_resource_value)
    else:
        cleaned['_resource_norm'] = 'unknown'

    cleaned['_amount_numeric'] = 0.0
    cleaned['_expense_numeric'] = 0.0

    for col in cleaned.columns:
        col_lower = col.lower()

        if 'amount' in col_lower:
            cleaned['_amount_numeric'] = pd.to_numeric(
                cleaned[col],
                errors='coerce'
            ).fillna(0.0)

        elif 'expense' in col_lower:
            cleaned['_expense_numeric'] = pd.to_numeric(
                cleaned[col],
                errors='coerce'
            ).fillna(0.0)

    return cleaned.reset_index(drop=True)


def learn_control_flow_profile(grouped, min_edge_support=0.05):
    edge_counts = {}
    activity_case_counts = {}
    activity_repetitions = {}
    total_cases = 0

    for _, group in grouped:
        total_cases += 1
        activities = list(group['_activity_norm'])

        for activity in set(activities):
            activity_case_counts[activity] = activity_case_counts.get(activity, 0) + 1

        counts = {}

        for activity in activities:
            counts[activity] = counts.get(activity, 0) + 1

        for activity, count in counts.items():
            activity_repetitions.setdefault(activity, []).append(count)

        for current_activity, next_activity in zip(activities, activities[1:]):
            edge = (current_activity, next_activity)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    min_edge_count = max(1, int(np.ceil(total_cases * min_edge_support)))

    frequent_edges = {
        edge
        for edge, count in edge_counts.items()
        if count >= min_edge_count
    }

    required_activities = {
        activity
        for activity, count in activity_case_counts.items()
        if count / max(total_cases, 1) >= 0.60
    }

    max_repetitions = {
        activity: max(1, int(np.ceil(np.percentile(counts, 95))))
        for activity, counts in activity_repetitions.items()
    }

    return {
        'frequent_edges': frequent_edges,
        'required_activities': required_activities,
        'max_repetitions': max_repetitions,
        'edge_counts': edge_counts,
        'total_cases': total_cases,
    }


def learn_resource_profile(profile_df):
    activity_resources = {}
    resource_counts = {}
    total_events = len(profile_df)

    for _, event in profile_df.iterrows():
        activity = event['_activity_norm']
        resource = event['_resource_norm']

        activity_resources.setdefault(activity, {})
        activity_resources[activity][resource] = (
            activity_resources[activity].get(resource, 0) + 1
        )
        resource_counts[resource] = resource_counts.get(resource, 0) + 1

    allowed_by_activity = {}

    for activity, counts in activity_resources.items():
        total = sum(counts.values())
        min_count = max(1, int(np.ceil(total * 0.01)))
        allowed_by_activity[activity] = {
            resource
            for resource, count in counts.items()
            if count >= min_count
        }

    resource_frequency_share = {
        resource: count / max(total_events, 1)
        for resource, count in resource_counts.items()
    }

    return {
        'allowed_by_activity': allowed_by_activity,
        'resource_frequency_share': resource_frequency_share,
    }


def replay_event_states(cleaned_df):
    digital_twin = DigitalTwin()

    for _, event in cleaned_df.sort_values('_parsed_timestamp').iterrows():
        timestamp = event['_parsed_timestamp']

        if pd.isna(timestamp):
            timestamp = None

        digital_twin.update_case_state({
            'case_id': event['_case_id_norm'],
            'activity': event['_activity_norm'],
            'resource': event['_resource_norm'],
            'timestamp': timestamp,
        })

    return digital_twin.case_states


def choose_detection_threshold(final_df, score_col='anomaly_score'):
    scores = final_df[score_col].astype(float)

    if (
        'label' in final_df.columns
        and final_df['label'].isin(['regular', 'deviant']).all()
        and final_df['label'].nunique() == 2
    ):
        y_true = final_df['label'].eq('deviant').astype(int)
        best_threshold = float(scores.min())
        best_f1 = -1.0

        for threshold in sorted(scores.unique()):
            y_pred = (scores >= threshold).astype(int)
            current_f1 = f1_score(y_true, y_pred, zero_division=0)

            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = float(threshold)

        return round(best_threshold, 4), f'{score_col}_label_calibrated_f1'

    unique_scores = np.sort(scores.unique())

    if len(unique_scores) == 1:
        return round(float(unique_scores[0]), 4), f'{score_col}_single_score'

    best_threshold = float(unique_scores[0])
    best_separation = -1.0
    total_count = len(scores)

    for threshold in unique_scores[1:]:
        lower_group = scores[scores < threshold]
        upper_group = scores[scores >= threshold]

        if lower_group.empty or upper_group.empty:
            continue

        separation = (
            len(lower_group)
            * len(upper_group)
            / (total_count ** 2)
            * (upper_group.mean() - lower_group.mean()) ** 2
        )

        if separation > best_separation:
            best_separation = separation
            best_threshold = float(threshold)

    return round(best_threshold, 4), f'{score_col}_score_distribution_otsu'


def build_case_features(cleaned_df, cf_profile, resource_profile, case_states, label_col):
    rows = []
    allowed_by_activity = resource_profile['allowed_by_activity']
    resource_frequency = resource_profile['resource_frequency_share']

    for case_id, group in cleaned_df.groupby('_case_id_norm', sort=False):
        group = group.sort_values('_parsed_timestamp')
        activities = list(group['_activity_norm'])
        resources = list(group['_resource_norm'])
        unique_resources = list(set(resources))

        activity_counts = {}

        for activity in activities:
            activity_counts[activity] = activity_counts.get(activity, 0) + 1

        frequent_edges = cf_profile['frequent_edges']
        edges = list(zip(activities, activities[1:]))

        seq_violations = sum(
            1
            for edge in edges
            if edge not in frequent_edges
        )

        wrong_order_ratio = (
            seq_violations / max(len(edges), 1)
            if edges else 0.0
        )

        missing_steps = len(
            cf_profile['required_activities'] - set(activities)
        )

        duplicate_steps = sum(
            max(
                0,
                count - cf_profile['max_repetitions'].get(activity, 1)
            )
            for activity, count in activity_counts.items()
        )

        timestamps = group['_parsed_timestamp'].dropna()
        total_hrs = 0.0
        max_step_hrs = 0.0
        std_step_hrs = 0.0

        if len(timestamps) > 1:
            total_hrs = (
                timestamps.max() - timestamps.min()
            ).total_seconds() / 3600

            diffs = timestamps.diff().dt.total_seconds().dropna() / 3600

            if len(diffs) > 0:
                max_step_hrs = float(diffs.max())
                std_step_hrs = float(0.0 if pd.isna(diffs.std()) else diffs.std())

        resource_frequency_in_case = {}

        for resource in resources:
            resource_frequency_in_case[resource] = (
                resource_frequency_in_case.get(resource, 0) + 1
            )

        max_resource_usage = (
            max(resource_frequency_in_case.values())
            if resource_frequency_in_case else 0
        )

        unusual_resource_events = 0

        for activity, resource in zip(activities, resources):
            allowed = allowed_by_activity.get(activity)

            if allowed is not None and resource not in allowed:
                unusual_resource_events += 1

        resource_rarity = 0.0

        if resources:
            resource_rarity = float(np.mean([
                1.0 - resource_frequency.get(resource, 0.0)
                for resource in resources
            ]))

        state = case_states.get(str(case_id), {})
        true_label = 'unknown'

        if label_col is not None:
            labels = group[label_col].astype(str).str.lower()
            true_label = 'deviant' if any(labels == 'deviant') else 'regular'

        res_n_resources = len(unique_resources)
        event_count = len(group)

        row = {
            'case_id': str(case_id),
            'label': true_label,
            'cf_n_events': event_count,
            'cf_seq_violations': seq_violations,
            'cf_missing_steps': missing_steps,
            'cf_duplicate_steps': duplicate_steps,
            'cf_wrong_order_ratio': wrong_order_ratio,
            'cf_has_appeal': int(any('appeal' in activity for activity in activities)),
            'cf_has_penalty': int(any('penalty' in activity for activity in activities)),
            'cf_has_payment': int(any('payment' in activity for activity in activities)),
            'temp_total_hrs': total_hrs,
            'temp_max_step_hrs': max_step_hrs,
            'temp_std_step_hrs': std_step_hrs,
            'res_n_resources': res_n_resources,
            'res_single_resource': int(res_n_resources == 1),
            'res_many_resources': res_n_resources,
            'res_dominant_resource_ratio': (
                max_resource_usage / max(len(resources), 1)
            ),
            'res_rpa_flag': int(any(
                any(keyword in resource for keyword in [
                    'bot',
                    'robot',
                    'system',
                    'auto',
                    'rpa'
                ])
                for resource in resources
            )),
            'res_unusual_activity_count': unusual_resource_events,
            'res_unusual_activity_ratio': (
                unusual_resource_events / max(event_count, 1)
            ),
            'res_workload_share': (
                max_resource_usage / max(event_count, 1)
            ),
            'res_resource_rarity': resource_rarity,
            'event_count': event_count,
            'dt_last_activity': state.get('last_activity', ''),
            'dt_execution_state': state.get('execution_state', 'running'),
            'amount': float(group['_amount_numeric'].mean()),
            'expense': float(group['_expense_numeric'].mean()),
        }

        rows.append(row)

    return pd.DataFrame(rows).fillna(0)


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


def train_and_detect(csv_path):
    df = pd.read_csv(
        csv_path,
        sep=None,
        engine='python'
    )

    columns = detect_event_log_columns(df)
    label_col = columns['label']

    cleaned_df = preprocess_event_log(df, columns)

    profile_df = cleaned_df

    if label_col is not None:
        labels_for_profile = cleaned_df[label_col].astype(str).str.lower()
        regular_df = cleaned_df[labels_for_profile == 'regular']

        if not regular_df.empty:
            profile_df = regular_df

    profile_grouped = profile_df.groupby('_case_id_norm', sort=False)

    cf_profile = learn_control_flow_profile(profile_grouped)
    resource_profile = learn_resource_profile(profile_df)
    case_states = replay_event_states(cleaned_df)

    feature_df = build_case_features(
        cleaned_df,
        cf_profile,
        resource_profile,
        case_states,
        label_col
    )

    train_feature_df = feature_df

    if 'label' in feature_df.columns:
        regular_feature_df = feature_df[feature_df['label'] == 'regular']

        if not regular_feature_df.empty:
            train_feature_df = regular_feature_df

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

    dt = DigitalTwin()
    dt.fit(train_feature_df, numeric_cols)
    dt.seed_states(case_states)

    ddc = DynamicDeclarativeConstraints()
    ddc.fit(
        dt,
        numeric_cols,
        cf_profile=cf_profile,
        resource_profile=resource_profile
    )

    mv_arm = MVARMiner()
    mv_arm.fit(train_feature_df)

    ib = IntelligentBody(dt, ddc, mv_arm)
    ib.calibrate_weights(feature_df, numeric_cols)

    precomputed_arm = mv_arm.score_dataframe(feature_df)
    scoring_df = feature_df.copy()
    scoring_df['_pre_arm_score'] = precomputed_arm['arm_score']
    scoring_df['_pre_arm_rules_hit'] = precomputed_arm['arm_rules_hit']
    scoring_df['_pre_violated_arm_rules'] = precomputed_arm['violated_arm_rules']

    if len(scoring_df) > 10000:
        result_df = ib.score_all_fast(scoring_df)
    else:
        result_df = ib.score_all(scoring_df)

    final_df = pd.merge(
        feature_df,
        result_df,
        on='case_id'
    )

    threshold, threshold_method = choose_detection_threshold(final_df)

    final_df['threshold'] = threshold
    final_df['threshold_method'] = threshold_method
    final_df['predicted_label'] = np.where(
        final_df['anomaly_score'] >= threshold,
        'deviant',
        'regular'
    )

    if 'risk_level' not in final_df.columns:
        final_df['risk_level'] = 'Low'

    final_df = add_lightweight_baselines(final_df, mv_arm)

    model_bundle = {
        'digital_twin': dt,
        'ddc': ddc,
        'mv_arm': mv_arm,
        'intelligent_body': ib,
        'control_flow_profile': cf_profile,
        'resource_profile': resource_profile,
        'numeric_cols': numeric_cols,
    }

    with open(MODEL_DIR / 'tf_model_bundle.pkl', 'wb') as f:
        pickle.dump(model_bundle, f)

    output_path = RESULT_DIR / 'prediction_results.csv'
    final_df.to_csv(output_path, index=False)

    anomaly_count = int(final_df['predicted_label'].eq('deviant').sum())
    normal_count = len(final_df) - anomaly_count

    return {
        'total_rows': len(final_df),
        'anomaly_count': anomaly_count,
        'normal_count': normal_count,
        'threshold': threshold,
        'threshold_method': threshold_method,
        'result_file': str(output_path.name)
    }
