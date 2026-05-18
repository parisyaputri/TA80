from flask import Flask, render_template, request, jsonify
from pathlib import Path
import pandas as pd
import sys

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from utils.evaluation import evaluate_model
from train_pipeline import train_and_detect
from utils.save_evaluation import save_evaluation

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_FOLDER = BASE_DIR / 'flask_app' / 'uploads'

UPLOAD_FOLDER.mkdir(exist_ok=True)


def clean_text(value, default=''):

    if pd.isna(value):
        return default

    text = str(value)

    if text.strip().lower() in ['nan', 'none', 'nat']:
        return default

    return text


def clean_float(value, default=0.0):

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default

    if pd.isna(number):
        return default

    return number


# =========================================================
# HOME
# =========================================================
@app.route('/')
def home():

    return render_template(
        'index.html'
    )


# =========================================================
# CSV UPLOAD + DETECTION
# =========================================================
@app.route('/classify-csv', methods=['POST'])
def upload_csv():
    print("\nUPLOAD CSV TRIGGERED")
    try:

        # ===== CHECK FILE =====
        if 'file' not in request.files:

            return jsonify({
                'error': 'No file uploaded'
            })

        file = request.files['file']

        if file.filename == '':

            return jsonify({
                'error': 'No file selected'
            })

        # ===== SAVE FILE =====
        filepath = (
            UPLOAD_FOLDER /
            file.filename
        )

        file.save(filepath)

        # ===== TRAIN & DETECT =====
        result = train_and_detect(filepath)
        print("\nTRAIN & DETECT FINISHED")

        # ===== LOAD RESULT CSV =====
        result_csv_path = (
            BASE_DIR /
            'dataOutput' /
            'results' /
            'prediction_results.csv'
        )

        df = pd.read_csv(
            result_csv_path,
            keep_default_na=False
        )

        # =================================================
        # EVALUATION
        # =================================================
        if 'label' in df.columns:

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

            print("\n===== DT-IB ADAPTIVE MODEL =====")

            save_evaluation(
                y_true,
                y_pred,
                y_scores,

                BASE_DIR /
                'dataOutput' /
                'results' /
                f'{Path(file.filename).stem}_eval.txt',

                'DT-IB ADAPTIVE MODEL'
            )

            if 'static_dc_predicted_label' in df.columns:

                print("\n===== BASELINE: STATIC DC =====")

                save_evaluation(
                    y_true,
                    df['static_dc_predicted_label'],

                    (
                        df['static_dc_score']
                        if 'static_dc_score' in df.columns
                        else None
                    ),

                    BASE_DIR /
                    'dataOutput' /
                    'results' /
                    f'{Path(file.filename).stem}_static_dc_eval.txt',

                    'BASELINE STATIC DC'
                )

            if 'single_arm_predicted_label' in df.columns:

                print("\n===== BASELINE: SINGLE-VIEW ARM =====")

                save_evaluation(
                    y_true,
                    df['single_arm_predicted_label'],

                    (
                        df['single_arm_score']
                        if 'single_arm_score' in df.columns
                        else None
                    ),

                    BASE_DIR /
                    'dataOutput' /
                    'results' /
                    f'{Path(file.filename).stem}_single_arm_eval.txt',

                    'BASELINE SINGLE-VIEW ARM'
                )

            else:

                print("\n===== EVALUATION SKIPPED =====")
                print("Column 'label' not found")
        # =================================================
        # FORMAT RESULTS
        # =================================================
        results = []

        for idx, row in df.iterrows():

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

        # =================================================
        # SUMMARY
        # =================================================
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

        # =================================================
        # RESPONSE
        # =================================================
        return jsonify({

            'summary': {

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
                    )
            },

            'results':
                results
        })

    except Exception as e:

        print('ERROR:', str(e))

        return jsonify({
            'error': str(e)
        })


# =========================================================
# RUN APP
# =========================================================
if __name__ == '__main__':

    app.run(
        host='127.0.0.1',
        port=8765,
        debug=False,
        use_reloader=False
    )
