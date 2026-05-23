# 🚦 Traffic Fine Anomaly Detection System

## 📋 Table of Contents

* [Project Overview](#-project-overview)
* [Features](#-features)
* [System Architecture](#️-system-architecture)
* [Folder Structure](#-folder-structure)
* [Dataset](#-dataset)
* [Getting Started](#-getting-started)
* [Running the Application](#️-running-the-application)
* [Application Workflow](#-application-workflow)
* [Model Pipeline](#-model-pipeline)
* [Output Files](#-output-files)
* [Evaluation Metrics](#-evaluation-metrics)
* [Technologies Used](#️-technologies-used)
* [Common Issues](#-common-issues)
* [Future Improvements](#-future-improvements)
* [License](#-license)

---

# 🔍 Project Overview

This project is a Flask-based anomaly detection system for business process event logs using the Traffic Fine Event Log dataset. The system detects anomalous process executions by combining multiple detection perspectives such as:

* Control-Flow anomalies
* Temporal anomalies
* Resource anomalies
* Business rule violations
* Multi-View Association Rule Mining (MV-ARM)

The application provides:

* CSV upload interface
* Automatic anomaly detection
* Evaluation metrics
* Exportable prediction results
* Human-readable explanations for detected anomalies

The system uses an adaptive weighted scoring mechanism where anomaly components contribute dynamically based on their AUC performance.

---

# ✨ Features

## Core Features

* Upload CSV event log datasets
* Automatic separator detection (`;` or `,`)
* Adaptive anomaly scoring
* ROC-Youden threshold optimization
* Risk level classification
* Evaluation metrics generation
* Export prediction results as CSV
* Flask web interface

## Detection Perspectives

### Control-Flow Analysis

Detects:

* Missing activities
* Sequence violations
* Duplicate activities
* Wrong execution order

### Temporal Analysis

Detects:

* Abnormal duration
* Time deviation
* Temporal outliers

### Resource Analysis

Detects:

* Resource dominance
* Too many resources involved
* Single resource dependency

### Business Rule Analysis

Detects violations based on predefined business constraints.

### MV-ARM Analysis

Uses Multi-View Association Rule Mining to identify unusual process behavior patterns.

---

# 🏗️ System Architecture

The project follows a modular Flask architecture:

```text
User Upload CSV
        ↓
Flask Routes
        ↓
Train & Detection Pipeline
        ↓
Feature Engineering
        ↓
Anomaly Detection Components
        ↓
Adaptive Scoring Fusion
        ↓
Prediction Results
        ↓
Evaluation & Visualization
```

---

# 📂 Folder Structure

```text
project-root/
├── app/
│   ├── services/
│   │   ├── format_results.py
│   │   ├── build_summary.py
│   │   └── run_evaluations.py
│   ├── templates/
│   │   └── index.html
│   ├── static/
│   │   ├── css/
│   │   └── js/
│   ├── routes.py                  # Flask routes
│   └── __init__.py                # Flask initialization
│
├── utils/
│   ├── evaluation.py              # Evaluation metrics
│   ├── result_handler.py          # Save outputs & evaluations
│   ├── anomaly_detection.py
│   ├── feature_engineering.py
│   └── preprocessing.py
│
├── outputs/
│   ├── results/
│   │   └── prediction_results.csv
│   ├── evaluations/
│   └── models/
│
├── uploads/                       # Uploaded CSV files
├── train_pipeline.py              # Main ML pipeline
├── run.py                         # Flask entry point
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 📊 Dataset

The system uses the Traffic Fine Event Log dataset.

Example attributes:

| Column             | Description                        |
| ------------------ | ---------------------------------- |
| Case ID            | Unique process instance            |
| Activity           | Executed activity                  |
| Resource           | Employee/system executing activity |
| Complete Timestamp | Event timestamp                    |
| amount             | Fine amount                        |
| expense            | Additional expenses                |
| points             | Penalty points                     |
| vehicleClass       | Vehicle category                   |
| label              | Ground truth anomaly label         |

---

# 🚀 Getting Started

## 1. Clone Repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

Run Flask server:

```bash
python run.py
```

Server will run at:

```text
http://127.0.0.1:8765
```

---

# 🌐 Application Workflow

## CSV Upload

Users upload event log CSV files through the web interface.

Supported separators:

* `,`
* `;`

Separator detection is automatic.

---

## Detection Pipeline

The uploaded dataset passes through:

1. Data preprocessing
2. Feature engineering
3. Control-flow analysis
4. Temporal analysis
5. Resource analysis
6. MV-ARM mining
7. Adaptive score fusion
8. Threshold optimization
9. Prediction generation

---

## Threshold Optimization

When ground-truth labels are available, the system splits case-level
results into deterministic calibration and holdout-test subsets based
on the uploaded data distribution.

* Threshold calibration uses ROC analysis and Youden's J statistic:
  `J = TPR - FPR`.
* The selected threshold is applied to all cases.
* Reported evaluation metrics use only the holdout-test subset, so the
  threshold is not optimized and evaluated on the exact same cases.
* The calibration/evaluation split is stratified by label and ordered
  from data-derived case keys, so it does not depend on a fixed random
  seed or hardcoded split ratio.
* If labels are unavailable, the system falls back to an unsupervised
  score-distribution threshold.
* If a score has no positive Youden gain on the calibration subset, the
  system marks the threshold method as `roc_youden_j_no_positive_gain`
  and avoids classifying every case as anomalous.

---

# 🧠 Model Pipeline

Main pipeline file:

```text
train_pipeline.py
```

Main responsibilities:

* Load dataset
* Build case-level features
* Generate anomaly scores
* Apply adaptive weighting
* Calculate ROC-Youden thresholds on calibration data
* Generate predictions
* Save outputs

---

# 📤 Output Files

Generated outputs:

## Prediction Results

```text
outputs/results/prediction_results.csv
```

Contains:

* Case ID
* Actual label
* Predicted label
* Evaluation split
* Anomaly score
* Risk level
* DDC score
* ARM score
* Threshold method
* Business rule score
* Explanations

---

## Evaluation Results

```text
outputs/evaluations/
```

Contains:

* Accuracy
* Precision
* Recall
* F1-score
* MCC
* FAR
* AUC-ROC
* AUC-PR
* Confusion Matrix

---

# 📈 Evaluation Metrics

The system evaluates performance using:

| Metric    | Description                             |
| --------- | --------------------------------------- |
| Accuracy  | Overall prediction correctness          |
| Precision | Positive prediction quality             |
| Recall    | Detection sensitivity                   |
| F1-Score  | Balance between precision & recall      |
| MCC       | Correlation between prediction & actual |
| FAR       | False alarm rate                        |
| AUC-ROC   | ROC area under curve                    |
| AUC-PR    | Precision-Recall area                   |

---

# 🛠️ Technologies Used

## Backend

* Flask
* Python
* Pandas
* NumPy
* Scikit-learn

## Data Processing

* Event log preprocessing
* Feature engineering
* Statistical anomaly detection
* Association Rule Mining

## Frontend

* HTML
* CSS
* JavaScript

---

# 🔧 Common Issues

## CSV Separator Problems

The application supports both:

```csv
column1,column2
```

and:

```csv
column1;column2
```

Separators are detected automatically.

---

## Flask App Not Running

Make sure virtual environment is activated:

```bash
venv\Scripts\activate
```

Then run:

```bash
python run.py
```

---

## Module Not Found Error

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 📝 Notes

* Prediction results are exported using `;` separator for better Excel compatibility.
* Uploaded files are stored temporarily inside `uploads/`.
* Model outputs are stored in `outputs/`.
* Generated `.pkl` model files are recommended to be excluded from Git.

---

# 👩‍💻 Developer Notes

Recommended `.gitignore` additions:

```gitignore
uploads/
outputs/
*.pkl
*.joblib
venv/
```

---

# 📄 License

This project is developed for academic and research purposes.
