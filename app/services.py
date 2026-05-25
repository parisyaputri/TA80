import pandas as pd
from utils.evaluation import evaluate_model

def clean_text(value, default=''):

    if pd.isna(value):
        return default

    text = str(value)

    if text.strip().lower() in [
        'nan',
        'none',
        'nat'
    ]:
        return default

    return text


def clean_float(value, default=0.0):

    try:
        number = float(value)

    except (
        TypeError,
        ValueError
    ):
        return default

    if pd.isna(number):
        return default

    return number


def run_evaluations(
    df,
    file,
    base_dir
):

    if 'label' not in df.columns:

        print("\n===== EVALUATION SKIPPED =====")
        print("Column 'label' not found")

        return

    y_true = df['label']

    if 'predicted_label' in df.columns:

        y_pred = df['predicted_label']

    else:

        y_pred = df['risk_level'].apply(

            lambda x:
                'deviant'
                if str(x).lower() == 'high'
                else 'regular'
        )

    y_scores = (
        df['anomaly_score']
        if 'anomaly_score' in df.columns
        else None
    )

    evaluate_model(y_true, y_pred, y_scores)

def format_results(df):

    results = []

    for _, row in df.iterrows():

        risk_level = clean_text(
            row.get(
                'risk_level',
                ''
            ),
            ''
        )

        is_anomaly = (
            str(
                row.get(
                    'predicted_label',
                    ''
                )
            ).lower()
            == 'deviant'
            or risk_level.lower() == 'high'
        )

        anomaly_score = clean_float(
            row.get(
                'anomaly_score',
                0
            )
        )

        predicted_label = clean_text(
            row.get(
                'predicted_label',
                (
                    'deviant'
                    if is_anomaly
                    else 'regular'
                )
            ),
            (
                'deviant'
                if is_anomaly
                else 'regular'
            )
        )

        risk_level = clean_text(
            risk_level,
            (
                'High'
                if is_anomaly
                else 'Low'
            )
        )

        normalized_score = round(

            min(
                max(
                    anomaly_score,
                    0
                ),
                1
            ),
            3
        )

        results.append({

            'case_id':
                clean_text(
                    row.get(
                        'case_id',
                        ''
                    )
                ),

            'true_label':
                clean_text(
                    row.get(
                        'label',
                        'unknown'
                    ),
                    'unknown'
                ),

            'predicted_label':
                predicted_label,

            'anomaly_score':
                normalized_score,

            'risk_level':
                risk_level,

            'badge_colour':
                'danger'
                if is_anomaly
                else 'success',

            'anomaly_types':
                clean_text(
                    row.get(
                        'anomaly_types',
                        'No anomaly'
                    ),
                    'No anomaly'
                ),

            'ddc_score':
                round(
                    clean_float(
                        row.get(
                            'ddc_score',
                            normalized_score
                        )
                    ),
                    3
                ),

            'z_score':
                round(
                    clean_float(
                        row.get(
                            'z_score',
                            normalized_score
                        )
                    ),
                    3
                ),

            'arm_score':
                round(
                    clean_float(
                        row.get(
                            'arm_score',
                            normalized_score
                        )
                    ),
                    3
                ),

            'br_score':
                round(
                    clean_float(
                        row.get(
                            'br_score',
                            normalized_score
                        )
                    ),
                    3
                ),

            'explanation':
                clean_text(
                    row.get(
                        'explanation',
                        (
                            'Unusual process flow detected'
                            if is_anomaly
                            else 'Normal process flow'
                        )
                    ),
                    (
                        'Unusual process flow detected'
                        if is_anomaly
                        else 'Normal process flow'
                    )
                )
        })

    return results


def build_summary(
    df,
    result
):

    total = len(df)

    predicted_counts = (
        df['predicted_label']
        .astype(str)
        .str.lower()
        .value_counts()
    )

    anomaly_count = int(
        predicted_counts.get(
            'deviant',
            0
        )
    )

    normal_count = int(
        predicted_counts.get(
            'regular',
            0
        )
    )

    risk_counts = (
        df['risk_level']
        .astype(str)
        .str.lower()
        .value_counts()
    )

    anomaly_pct = round(

        (
            anomaly_count / total
        ) * 100,

        2

    ) if total > 0 else 0

    return {

        'total':
            total,

        'deviant':
            anomaly_count,

        'regular':
            normal_count,

        'deviant_pct':
            anomaly_pct,

        'high_risk':
            int(
                risk_counts.get(
                    'high',
                    0
                )
            ),

        'medium_risk':
            int(
                risk_counts.get(
                    'medium',
                    0
                )
            ),

        'low_risk':
            int(
                risk_counts.get(
                    'low',
                    0
                )
            ),

        'threshold':
            clean_float(
                df['threshold'].iloc[0]
                if 'threshold' in df.columns
                and len(df) > 0
                else result['threshold']
            ),

        'threshold_method':
            clean_text(
                df['threshold_method'].iloc[0]
                if 'threshold_method' in df.columns
                and len(df) > 0
                else result.get(
                    'threshold_method',
                    'unknown'
                ),
                'unknown'
            ),

        'threshold_scope':
            clean_text(
                df['threshold_scope'].iloc[0]
                if 'threshold_scope' in df.columns
                and len(df) > 0
                else result.get(
                    'threshold_scope',
                    'unknown'
                ),
                'unknown'
            )
    }
