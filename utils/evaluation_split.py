import pandas as pd
from pandas.util import hash_pandas_object


CALIBRATION_SPLIT = 'baseline_train'
HOLDOUT_SPLIT = 'holdout_test'
FULL_DATASET_SPLIT = 'threshold_calibration_and_evaluation'
UNLABELED_SPLIT = 'unlabeled'
DEFAULT_TRAIN_FRACTION = 0.80


def has_binary_regular_deviant_labels(df, label_col='label'):
    return (
        label_col in df.columns
        and df[label_col].isin(['regular', 'deviant']).all()
        and df[label_col].nunique() == 2
    )


def build_case_train_evaluation_split(
    df,
    case_col='case_id',
    train_fraction=DEFAULT_TRAIN_FRACTION,
    split_col='evaluation_split'
):
    df[split_col] = UNLABELED_SPLIT

    if len(df) < 2:
        all_mask = pd.Series(
            True,
            index=df.index
        )

        return all_mask, all_mask, False

    train_mask = pd.Series(
        False,
        index=df.index
    )

    test_mask = pd.Series(
        False,
        index=df.index
    )

    case_ids = _ordered_case_ids(df, case_col)
    split_position = int(len(case_ids) * train_fraction)
    split_position = max(1, min(split_position, len(case_ids) - 1))

    train_cases = set(case_ids[:split_position])
    test_cases = set(case_ids[split_position:])

    case_values = _case_values(df, case_col)

    train_mask = case_values.isin(train_cases)
    test_mask = case_values.isin(test_cases)

    df.loc[
        train_mask,
        split_col
    ] = CALIBRATION_SPLIT

    df.loc[
        test_mask,
        split_col
    ] = HOLDOUT_SPLIT

    return train_mask, test_mask, True


def build_threshold_evaluation_split(
    final_df,
    label_col='label',
    train_fraction=DEFAULT_TRAIN_FRACTION
):
    return build_case_train_evaluation_split(
        final_df,
        case_col='case_id',
        train_fraction=train_fraction,
        split_col='evaluation_split'
    )


def threshold_scope(has_holdout_test):
    if has_holdout_test:
        return 'baseline_train_set'

    return 'full_dataset'


def evaluation_scope(has_holdout_test):
    if has_holdout_test:
        return 'HOLDOUT TEST 20%'

    return 'FULL DATASET'


def _case_values(df, case_col):
    if case_col in df.columns:
        return df[case_col].astype(str)

    return df.index.to_series().astype(str)


def _ordered_case_ids(df, case_col):
    case_values = _case_values(df, case_col)
    unique_cases = pd.DataFrame({
        'case_id': case_values.drop_duplicates()
    })

    row_key = unique_cases['case_id'].astype(str)
    hashed_key = hash_pandas_object(
        row_key,
        index=False
    )

    return (
        pd.Series(
            hashed_key.to_numpy(),
            index=unique_cases['case_id']
        )
        .sort_values(kind='mergesort')
        .index
        .tolist()
    )


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
