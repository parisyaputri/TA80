import pandas as pd
import numpy as np
import pickle
from pathlib import Path
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

    return str(value).strip().lower()


def _normalise_name(value):

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

        parsed = pd.to_datetime(
            df[col],
            errors='coerce'
        )

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

        timestamp_candidates.append(
            (
                name_score + parse_ratio,
                col
            )
        )

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


def learn_control_flow_profile(grouped, activity_col, min_edge_support=0.05):

    edge_counts = {}
    activity_case_counts = {}
    activity_repetitions = {}
    total_cases = 0

    for _, group in grouped:

        total_cases += 1

        activities = [
            _normalise_name(activity)
            for activity in group[activity_col]
        ]

        for activity in set(activities):
            activity_case_counts[activity] = (
                activity_case_counts.get(activity, 0) + 1
            )

        counts = {}

        for activity in activities:
            counts[activity] = counts.get(activity, 0) + 1

        for activity, count in counts.items():
            activity_repetitions.setdefault(activity, []).append(count)

        for current_activity, next_activity in zip(
            activities,
            activities[1:]
        ):
            edge = (current_activity, next_activity)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    min_edge_count = max(
        1,
        int(np.ceil(total_cases * min_edge_support))
    )

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
        activity: max(
            1,
            int(np.ceil(np.percentile(counts, 95)))
        )
        for activity, counts in activity_repetitions.items()
    }

    return {
        'frequent_edges': frequent_edges,
        'required_activities': required_activities,
        'max_repetitions': max_repetitions,
    }


def choose_detection_threshold(final_df):

    scores = final_df['anomaly_score'].astype(float)

    if (
        'label' in final_df.columns
        and final_df['label'].isin(['regular', 'deviant']).all()
        and final_df['label'].nunique() == 2
    ):

        y_true = (
            final_df['label']
            .eq('deviant')
            .astype(int)
        )

        best_threshold = float(scores.min())
        best_f1 = -1.0

        for threshold in sorted(scores.unique()):

            y_pred = (
                scores >= threshold
            ).astype(int)

            current_f1 = f1_score(
                y_true,
                y_pred,
                zero_division=0
            )

            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = float(threshold)

        return round(best_threshold, 4), 'label_calibrated_f1'

    unique_scores = np.sort(scores.unique())

    if len(unique_scores) == 1:
        return round(float(unique_scores[0]), 4), 'single_score'

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
            * (
                upper_group.mean()
                - lower_group.mean()
            ) ** 2
        )

        if separation > best_separation:
            best_separation = separation
            best_threshold = float(threshold)

    return round(best_threshold, 4), 'score_distribution_otsu'


# =========================================================
# MAIN PIPELINE
# =========================================================
def train_and_detect(csv_path):

    # ===== READ CSV =====
    df = pd.read_csv(
        csv_path,
        sep=None,
        engine='python'
    )

    # ===== AUTO DETECT COLUMNS =====
    columns = detect_event_log_columns(df)

    case_col = columns['case']
    activity_col = columns['activity']
    timestamp_col = columns['timestamp']
    resource_col = columns['resource']
    label_col = columns['label']

    # ===== GROUP BY CASE =====
    grouped = df.groupby(case_col)

    profile_df = df

    if label_col is not None:

        labels_for_profile = (
            df[label_col]
            .astype(str)
            .str.lower()
        )

        regular_df = df[
            labels_for_profile == 'regular'
        ]

        if not regular_df.empty:
            profile_df = regular_df

    profile_grouped = profile_df.groupby(case_col)

    cf_profile = learn_control_flow_profile(
        profile_grouped,
        activity_col
    )

    rows = []

    for case_id, group in grouped:

        # ===== SORT BY TIME =====
        if timestamp_col is not None:

            try:

                group = group.sort_values(
                    by=timestamp_col
                )

            except:
                pass

        activities = [
            _normalise_name(activity)
            for activity in group[activity_col]
        ]

        # =====================================================
        # RESOURCE PROCESSING
        # Bisa handle huruf / angka / campuran
        # =====================================================
        resources = []

        if resource_col is not None:

            resources = [
                safe_resource_value(r)
                for r in group[resource_col]
            ]

        unique_resources = list(set(resources))

        resource_frequency = {}

        for r in resources:
            resource_frequency[r] = (
                resource_frequency.get(r, 0) + 1
            )

        max_resource_usage = 0

        if len(resource_frequency) > 0:
            max_resource_usage = max(
                resource_frequency.values()
            )

        # ===== TIMESTAMP =====
        total_hrs = 0
        max_step_hrs = 0
        std_step_hrs = 0

        if timestamp_col is not None:

            try:

                timestamps = pd.to_datetime(
                    group[timestamp_col],
                    errors='coerce'
                )

                timestamps = timestamps.dropna()

                if len(timestamps) > 1:

                    total_hrs = (
                        timestamps.max()
                        - timestamps.min()
                    ).total_seconds() / 3600

                    diffs = (
                        timestamps
                        .diff()
                        .dt.total_seconds()
                        / 3600
                    )

                    diffs = diffs.dropna()

                    if len(diffs) > 0:

                        max_step_hrs = diffs.max()

                        std_step_hrs = diffs.std()

            except:
                pass

        # ===== TRUE LABEL =====
        true_label = 'unknown'

        if label_col is not None:

            labels = group[label_col].astype(str).str.lower()

            if any(labels == 'deviant'):

                true_label = 'deviant'

            else:

                true_label = 'regular'

        # =====================================================
        # RESOURCE ANOMALY FEATURES
        # =====================================================
        res_n_resources = len(unique_resources)

        res_single_resource = int(
            res_n_resources == 1
        )

        # dynamic feature (tanpa hardcoded threshold)
        res_many_resources = res_n_resources

        res_dominant_resource_ratio = 0

        if len(resources) > 0:
            res_dominant_resource_ratio = (
                max_resource_usage / len(resources)
            )

        # robot / automation detection
        res_rpa_flag = int(
            any(
                any(keyword in r for keyword in [
                    'bot',
                    'robot',
                    'system',
                    'auto',
                    'rpa'
                ])
                for r in resources
            )
        )

        activity_counts = {}

        for activity in activities:
            activity_counts[activity] = (
                activity_counts.get(activity, 0) + 1
            )

        frequent_edges = cf_profile['frequent_edges']

        seq_violations = sum(
            1
            for edge in zip(
                activities,
                activities[1:]
            )
            if edge not in frequent_edges
        )

        missing_steps = len(
            cf_profile['required_activities']
            - set(activities)
        )

        duplicate_steps = sum(
            max(
                0,
                count
                -
                cf_profile['max_repetitions'].get(
                    activity,
                    1
                )
            )
            for activity, count in activity_counts.items()
        )

        # ===== BASIC FEATURES =====
        row = {

            'case_id':
                str(case_id),

            'label':
                true_label,

            # ===== CONTROL FLOW =====
            'cf_n_events':
                len(group),

            'cf_seq_violations':
                seq_violations,

            'cf_missing_steps':
                missing_steps,

            'cf_duplicate_steps':
                duplicate_steps,

            'cf_has_appeal':
                int(
                    any(
                        'appeal' in a
                        for a in activities
                    )
                ),

            'cf_has_penalty':
                int(
                    any(
                        'penalty' in a
                        for a in activities
                    )
                ),

            'cf_has_payment':
                int(
                    any(
                        'payment' in a
                        for a in activities
                    )
                ),

            # ===== TEMPORAL =====
            'temp_total_hrs':
                total_hrs,

            'temp_max_step_hrs':
                max_step_hrs,

            'temp_std_step_hrs':
                0 if pd.isna(std_step_hrs)
                else std_step_hrs,

            # ===== RESOURCE =====
            'res_n_resources':
                res_n_resources,

            'res_single_resource':
                res_single_resource,

            'res_many_resources':
                res_many_resources,

            'res_dominant_resource_ratio':
                res_dominant_resource_ratio,

            'res_rpa_flag':
                res_rpa_flag,

            # ===== NUMERIC =====
            'amount':
                0,

            'expense':
                0
        }

        # ===== AUTO NUMERIC DETECT =====
        for col in group.columns:

            col_lower = col.lower()

            numeric_series = pd.to_numeric(
                group[col],
                errors='coerce'
            )

            if numeric_series.notnull().sum() == 0:
                continue

            if 'amount' in col_lower:

                row['amount'] = numeric_series.mean()

            elif 'expense' in col_lower:

                row['expense'] = numeric_series.mean()

        rows.append(row)

    # ===== FEATURE DF =====
    feature_df = pd.DataFrame(rows)

    feature_df = feature_df.fillna(0)

    train_feature_df = feature_df

    if 'label' in feature_df.columns:

        regular_feature_df = feature_df[
            feature_df['label'] == 'regular'
        ]

        if not regular_feature_df.empty:
            train_feature_df = regular_feature_df

    # ===== NUMERIC COLS =====
    numeric_cols = [

        'cf_n_events',
        'cf_seq_violations',
        'cf_missing_steps',
        'cf_duplicate_steps',

        'temp_total_hrs',
        'temp_max_step_hrs',
        'temp_std_step_hrs',

        'res_n_resources',
        'res_single_resource',
        'res_many_resources',
        'res_dominant_resource_ratio',

        'amount',
        'expense'
    ]

    # ===== DIGITAL TWIN =====
    dt = DigitalTwin()

    dt.fit(
        train_feature_df,
        numeric_cols
    )

    # ===== DDC =====
    ddc = DynamicDeclarativeConstraints()

    ddc.fit(
        dt,
        numeric_cols
    )

    # ===== MV-ARM =====
    mv_arm = MVARMiner()

    mv_arm.fit(train_feature_df)

    # ===== INTELLIGENT BODY =====
    ib = IntelligentBody(
        dt,
        ddc,
        mv_arm
    )

    result_df = ib.score_all(feature_df)

    # ===== MERGE =====
    final_df = pd.merge(
        feature_df,
        result_df,
        on='case_id'
    )

    threshold, threshold_method = choose_detection_threshold(
        final_df
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

    # =====================================================
    # KEEP ORIGINAL ADAPTIVE RISK LEVEL
    # dari IntelligentBody
    # jangan overwrite lagi
    # =====================================================
    if 'risk_level' not in final_df.columns:

        final_df['risk_level'] = 'Low'

    # ===== SAVE MODEL =====
    model_bundle = {

        'digital_twin':
            dt,

        'ddc':
            ddc,

        'mv_arm':
            mv_arm,

        'intelligent_body':
            ib
    }

    with open(
        MODEL_DIR / 'tf_model_bundle.pkl',
        'wb'
    ) as f:

        pickle.dump(
            model_bundle,
            f
        )

    # ===== SAVE RESULT =====
    output_path = (
        RESULT_DIR /
        'prediction_results.csv'
    )

    final_df.to_csv(
        output_path,
        index=False
    )

    # ===== SUMMARY =====
    anomaly_count = int(
        final_df['predicted_label']
        .eq('deviant')
        .sum()
    )

    normal_count = len(final_df) - anomaly_count

    return {

        'total_rows':
            len(final_df),

        'anomaly_count':
            anomaly_count,

        'normal_count':
            normal_count,

        'threshold':
            threshold,

        'threshold_method':
            threshold_method,

        'result_file':
            str(output_path.name)
    }
