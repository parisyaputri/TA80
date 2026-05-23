import pandas as pd
from pandas.util import hash_pandas_object


CALIBRATION_SPLIT = 'threshold_calibration'
HOLDOUT_SPLIT = 'holdout_test'
FULL_DATASET_SPLIT = 'threshold_calibration_and_evaluation'
UNLABELED_SPLIT = 'unlabeled'


def has_binary_regular_deviant_labels(df, label_col='label'):
    return (
        label_col in df.columns
        and df[label_col].isin(['regular', 'deviant']).all()
        and df[label_col].nunique() == 2
    )


def build_threshold_evaluation_split(final_df, label_col='label'):
    final_df['evaluation_split'] = UNLABELED_SPLIT

    if not has_binary_regular_deviant_labels(final_df, label_col):
        all_mask = pd.Series(
            True,
            index=final_df.index
        )

        return all_mask, all_mask, False

    label_counts = final_df[label_col].value_counts()

    if label_counts.min() < len(label_counts):
        final_df['evaluation_split'] = FULL_DATASET_SPLIT

        all_mask = pd.Series(
            True,
            index=final_df.index
        )

        return all_mask, all_mask, False

    calibration_mask = pd.Series(
        False,
        index=final_df.index
    )

    test_mask = pd.Series(
        False,
        index=final_df.index
    )

    for _, group in final_df.groupby(label_col, sort=False):
        ordered_index = _data_driven_order(group)
        split_position = len(ordered_index) // len(label_counts)

        calibration_index = ordered_index[:split_position]
        test_index = ordered_index[split_position:]

        calibration_mask.loc[calibration_index] = True
        test_mask.loc[test_index] = True

    final_df.loc[
        calibration_mask,
        'evaluation_split'
    ] = CALIBRATION_SPLIT

    final_df.loc[
        test_mask,
        'evaluation_split'
    ] = HOLDOUT_SPLIT

    return calibration_mask, test_mask, True


def threshold_scope(has_holdout_test):
    if has_holdout_test:
        return 'calibration_set'

    return 'full_dataset'


def evaluation_scope(has_holdout_test):
    if has_holdout_test:
        return 'HOLDOUT TEST'

    return 'FULL DATASET'


def _data_driven_order(group):
    if 'case_id' in group.columns:
        base_key = group['case_id'].astype(str)
    else:
        base_key = group.index.to_series().astype(str)

    row_key = (
        base_key
        + '|'
        + group.index.to_series().astype(str)
    )

    hashed_key = hash_pandas_object(
        row_key,
        index=False
    )

    return (
        pd.Series(
            hashed_key.to_numpy(),
            index=group.index
        )
        .sort_values(kind='mergesort')
        .index
    )
