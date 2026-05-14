import pandas as pd
import numpy as np
import pickle
from pathlib import Path

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


def train_and_detect(csv_path):

    # ===== READ CSV =====
    df = pd.read_csv(
        csv_path,
        sep=None,
        engine='python'
    )

    # ===== AUTO DETECT COLUMNS =====
    case_col = None
    activity_col = None
    timestamp_col = None

    for col in df.columns:

        col_lower = col.lower()

        # ===== CASE ID =====
        if (
            'case' in col_lower
            and (
                'id' in col_lower
                or 'concept:name' in col_lower
            )
        ):
            case_col = col

        # ===== ACTIVITY =====
        elif (
            'activity' in col_lower
            or 'concept:name' in col_lower
        ):
            activity_col = col

        # ===== TIMESTAMP =====
        if (
            'timestamp' in col_lower
            or 'time' in col_lower
            or 'date' in col_lower
        ):
            timestamp_col = col

    # ===== FALLBACK =====
    if case_col is None:
        case_col = df.columns[0]

    if activity_col is None:
        activity_col = df.columns[1]

    # ===== LABEL COLUMN DETECTION =====
    label_col = None

    for col in df.columns:

        col_lower = col.lower()

        if 'label' in col_lower:

            label_col = col
            break

    # ===== GROUP BY CASE =====
    grouped = df.groupby(case_col)

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

        activities = list(
            group[activity_col].astype(str)
        )

        # ===== TIMESTAMP =====
        total_hrs = 0
        max_step_hrs = 0
        std_step_hrs = 0

        if timestamp_col is not None:

            try:

                timestamps = pd.to_datetime(
                    group[timestamp_col]
                )

                total_hrs = (
                    timestamps.max()
                    - timestamps.min()
                ).total_seconds() / 3600

                diffs = timestamps.diff().dt.total_seconds() / 3600

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
                max(0, len(activities) - len(set(activities))),

            'cf_missing_steps':
                0,

            'cf_duplicate_steps':
                len(activities) - len(set(activities)),

            'cf_has_appeal':
                int(
                    any(
                        'appeal' in a.lower()
                        for a in activities
                    )
                ),

            'cf_has_penalty':
                int(
                    any(
                        'penalty' in a.lower()
                        for a in activities
                    )
                ),

            'cf_has_payment':
                int(
                    any(
                        'payment' in a.lower()
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
                1,

            'res_rpa_flag':
                0,

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

        'amount',
        'expense'
    ]

    # ===== DIGITAL TWIN =====
    dt = DigitalTwin()

    dt.fit(
        feature_df,
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

    mv_arm.fit(feature_df)

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
        (
            final_df['risk_level']
            == 'High'
        ).sum()
    )

    normal_count = len(final_df) - anomaly_count

    threshold = round(
        final_df['anomaly_score'].quantile(0.70),
        3
    )

    return {

        'total_rows':
            len(final_df),

        'anomaly_count':
            anomaly_count,

        'normal_count':
            normal_count,

        'threshold':
            threshold,

        'result_file':
            str(output_path.name)
    }