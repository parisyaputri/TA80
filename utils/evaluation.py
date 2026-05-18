from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef
)

import pandas as pd
import numpy as np


def evaluate_model(
    y_true,
    y_pred,
    y_scores=None
):

    # =========================
    # BASIC METRICS
    # =========================
    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        pos_label="deviant",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        pos_label="deviant",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label="deviant",
        zero_division=0
    )

    # =========================
    # CONFUSION MATRIX
    # =========================
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=["regular", "deviant"]
    )

    tn, fp, fn, tp = cm.ravel()

    # =========================
    # FAR
    # =========================
    far = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    # =========================
    # MCC
    # =========================
    mcc = matthews_corrcoef(
        y_true,
        y_pred
    )

    # =========================
    # AUC METRICS
    # =========================
    auc_roc = None
    auc_pr = None

    if y_scores is not None:

        y_true_binary = [
            1 if y == "deviant"
            else 0
            for y in y_true
        ]

        try:

            auc_roc = roc_auc_score(
                y_true_binary,
                y_scores
            )

            auc_pr = average_precision_score(
                y_true_binary,
                y_scores
            )

        except ValueError:

            auc_roc = None
            auc_pr = None

    # =========================
    # PRINT RESULTS
    # =========================
    print("\n===== EVALUATION RESULTS =====")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"FAR       : {far:.4f}")
    print(f"MCC       : {mcc:.4f}")

    if auc_roc is not None:
        print(f"AUC-ROC   : {auc_roc:.4f}")

    if auc_pr is not None:
        print(f"AUC-PR    : {auc_pr:.4f}")

    # =========================
    # CLASSIFICATION REPORT
    # =========================
    print("\n===== CLASSIFICATION REPORT =====")

    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    )

    # =========================
    # CONFUSION MATRIX TABLE
    # =========================
    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual Regular",
            "Actual Deviant"
        ],
        columns=[
            "Predicted Regular",
            "Predicted Deviant"
        ]
    )

    print("\n===== CONFUSION MATRIX =====")

    print(cm_df)
