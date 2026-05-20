import numpy as np
from sklearn.metrics import roc_curve

def choose_detection_threshold(final_df, score_col='anomaly_score'):
    scores = final_df[score_col].astype(float)

    if (
        'label' in final_df.columns
        and final_df['label'].isin(['regular', 'deviant']).all()
        and final_df['label'].nunique() == 2
    ):
        y_true = final_df['label'].eq('deviant').astype(int)
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        youden_j = tpr - fpr

        finite_mask = np.isfinite(thresholds)

        if not finite_mask.any():
            best_threshold = float(scores.min())
        else:
            finite_thresholds = thresholds[finite_mask]
            finite_youden_j = youden_j[finite_mask]
            best_index = int(np.argmax(finite_youden_j))
            best_threshold = float(finite_thresholds[best_index])

        return round(best_threshold, 4), f'{score_col}_roc_youden_j'

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

