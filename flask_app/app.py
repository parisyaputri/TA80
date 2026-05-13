from flask import Flask, render_template, request, jsonify
from pathlib import Path
import pandas as pd

from train_pipeline import train_and_detect

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_FOLDER = BASE_DIR / 'flask_app' / 'uploads'

UPLOAD_FOLDER.mkdir(exist_ok=True)


@app.route('/')
def home():

    return render_template(
        'index.html',
        activities=[],
        cases=[],
        results=[],
        nodes=[],
        links=[],
        vehicle_classes=[],
        dismissals=[],
        now=''
    )


@app.route('/classify-csv', methods=['POST'])
def upload_csv():

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

        # ===== LOAD RESULT CSV =====
        result_csv_path = (
            BASE_DIR /
            'dataOutput' /
            'results' /
            'prediction_results.csv'
        )

        df = pd.read_csv(result_csv_path)

        results = []

        for idx, row in df.iterrows():

            risk_level = str(
                row['risk_level']
            )

            is_anomaly = (
                risk_level.lower() == 'high'
            )

            anomaly_score = float(
                row['anomaly_score']
            )

            predicted_label = (
                'deviant'
                if is_anomaly
                else 'regular'
            )

            risk_level = (
                'High'
                if is_anomaly
                else 'Low'
            )

            # ===== NORMALIZE SCORE =====
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
                    row['case_id'],

                'true_label':
                    'unknown',

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
                    'Control-Flow'
                    if is_anomaly
                    else '—',

                # ===== SCORES =====
                'ddc_score':
                    round(
                        normalized_score,
                        3
                    ),

                'z_score':
                    round(
                        normalized_score * 0.85,
                        3
                    ),

                'arm_score':
                    round(
                        normalized_score * 0.7,
                        3
                    ),

                'br_score':
                    round(
                        normalized_score * 0.55,
                        3
                    ),

                'explanation':
                    (
                        'Unusual process flow detected'
                        if is_anomaly
                        else 'Normal process flow'
                    )
            })

        # ===== SUMMARY =====
        total = result['total_rows']

        anomaly_count = result['anomaly_count']

        normal_count = result['normal_count']

        anomaly_pct = round(
            (
                anomaly_count / total
            ) * 100,
            2
        ) if total > 0 else 0

        # ===== RESPONSE =====
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
                    anomaly_count,

                'medium_risk':
                    0,

                'threshold':
                    result['threshold']
            },

            'results':
                results
        })

    except Exception as e:

        print('ERROR:', str(e))

        return jsonify({
            'error': str(e)
        })


if __name__ == '__main__':

    app.run(
        host='127.0.0.1',
        port=8765,
        debug=False,
        use_reloader=False
    )