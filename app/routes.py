# ============================================
# FILE: app/routes.py
# ============================================

from pathlib import Path

import pandas as pd

from flask import (
    render_template,
    request,
    jsonify
)

from app.__init__ import flask_app

from train_pipeline import (
    train_and_detect
)

from app.services import (
    format_results,
    build_summary,
    run_evaluations
)


BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_FOLDER = BASE_DIR / 'uploads'

UPLOAD_FOLDER.mkdir(
    exist_ok=True
)


# ============================================
# HOME
# ============================================

@flask_app.route('/')
def home():

    return render_template(
        'index.html'
    )


# ============================================
# CSV CLASSIFICATION
# ============================================

@flask_app.route(
    '/classify-csv',
    methods=['POST']
)
def upload_csv():

    print("\nUPLOAD CSV TRIGGERED")

    try:

        # ====================================
        # VALIDATE FILE
        # ====================================

        if 'file' not in request.files:

            return jsonify({
                'error': 'No file uploaded'
            })

        file = request.files['file']

        if file.filename == '':

            return jsonify({
                'error': 'No file selected'
            })

        # ====================================
        # SAVE FILE
        # ====================================

        filepath = (
            UPLOAD_FOLDER /
            file.filename
        )

        file.save(filepath)

        # ====================================
        # TRAIN + DETECT
        # ====================================

        result = train_and_detect(
            filepath
        )

        print("\nTRAIN & DETECT FINISHED")

        # ====================================
        # LOAD RESULT CSV
        # ====================================

        result_csv_path = (
            BASE_DIR /
            'outputs' /
            'results' /
            'prediction_results.csv'
        )

        df = pd.read_csv(
            result_csv_path,
            keep_default_na=False
        )

        # ====================================
        # EVALUATION
        # ====================================

        run_evaluations(
            df,
            file,
            BASE_DIR
        )

        # ====================================
        # FORMAT RESPONSE
        # ====================================

        results = format_results(df)

        summary = build_summary(
            df,
            result
        )

        # ====================================
        # RESPONSE
        # ====================================

        return jsonify({

            'summary': summary,

            'results': results
        })

    except Exception as e:

        print(
            'ERROR:',
            str(e)
        )

        return jsonify({
            'error': str(e)
        })