from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import pandas as pd


def evaluate_model(y_true, y_pred):

    # =========================
    # METRICS
    # =========================
    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        pos_label="deviant"
    )

    recall = recall_score(
        y_true,
        y_pred,
        pos_label="deviant"
    )

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label="deviant"
    )

    # =========================
    # PRINT RESULTS
    # =========================
    print("\n===== EVALUATION RESULTS =====")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")

    # =========================
    # CLASSIFICATION REPORT
    # =========================
    print("\n===== CLASSIFICATION REPORT =====")

    print(classification_report(y_true, y_pred))

    # =========================
    # CONFUSION MATRIX
    # =========================
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=["regular", "deviant"]
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Regular", "Actual Deviant"],
        columns=["Predicted Regular", "Predicted Deviant"]
    )

    print("\n===== CONFUSION MATRIX =====")

    print(cm_df)