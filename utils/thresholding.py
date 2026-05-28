import numpy as np
from sklearn.metrics import roc_curve

NO_POSITIVE_GAIN_SUFFIX = 'no_positive_gain'


def choose_detection_threshold(final_df, score_col='anomaly_score'):
    scores = final_df[score_col].astype(float)


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


def apply_detection_threshold(
    df,
    score_col,
    threshold,
    threshold_method
):
    if str(threshold_method).endswith(NO_POSITIVE_GAIN_SUFFIX):
        return np.full(
            len(df),
            'regular',
            dtype=object
        )

    return np.where(
        df[score_col].astype(float) >= threshold,
        'deviant',
        'regular'
    )

