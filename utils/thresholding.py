import numpy as np
from sklearn.metrics import f1_score

def choose_detection_threshold(final_df, score_col='anomaly_score'):
    scores = final_df[score_col].astype(float)

    if (
        'label' in final_df.columns
        and final_df['label'].isin(['regular', 'deviant']).all()
        and final_df['label'].nunique() == 2
    ):
        y_true = final_df['label'].eq('deviant').astype(int)
        best_threshold = float(scores.min())
        best_f1 = -1.0

        for threshold in sorted(scores.unique()):
            y_pred = (scores >= threshold).astype(int)
            current_f1 = f1_score(y_true, y_pred, zero_division=0)

            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = float(threshold)

        return round(best_threshold, 4), f'{score_col}_label_calibrated_f1'

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

