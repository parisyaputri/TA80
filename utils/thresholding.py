import numpy as np
from sklearn.metrics import roc_curve

NO_POSITIVE_GAIN_SUFFIX = 'no_positive_gain'


def choose_detection_threshold(final_df, score_col='anomaly_score'):
    scores = final_df[score_col].astype(float)

    if (
        'label' in final_df.columns
        and final_df['label'].isin(['regular', 'deviant']).all()
        and final_df['label'].nunique() == 2
    ):
        y_true = final_df['label'].eq('deviant').astype(int)
        
        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        
        # Calculate F1 score for each threshold
        denom = precision + recall
        f1_scores = np.divide(
            2 * precision * recall,
            denom,
            out=np.zeros_like(denom),
            where=denom > 0
        )
        
        # Slice to match the length of thresholds (precision/recall have N+1 elements)
        f1_scores = f1_scores[:len(thresholds)]
        
        if len(thresholds) > 0:
            best_idx = int(np.argmax(f1_scores))
            best_threshold = float(thresholds[best_idx])
            
            if f1_scores[best_idx] <= 0:
                best_threshold = float(scores.max())
                method_suffix = f'pr_max_f1_{NO_POSITIVE_GAIN_SUFFIX}'
            else:
                method_suffix = 'pr_max_f1'
        else:
            best_threshold = float(scores.max())
            method_suffix = f'pr_max_f1_{NO_POSITIVE_GAIN_SUFFIX}'

        return round(best_threshold, 4), f'{score_col}_{method_suffix}'

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

