DEFAULT_BASELINE_KEEP_FRACTION = 0.80
DEFAULT_BASELINE_THRESHOLD_QUANTILE = 0.95


def select_stable_baseline_cases(
    scored_df,
    score_col='anomaly_score',
    case_col='case_id',
    keep_fraction=DEFAULT_BASELINE_KEEP_FRACTION
):
    if scored_df.empty:
        return set()

    keep_count = int(len(scored_df) * keep_fraction)
    keep_count = max(1, min(keep_count, len(scored_df)))

    ranked_df = scored_df.copy()
    rank_score = ranked_df[score_col].astype(float)

    if 'cf_has_payment' in ranked_df.columns:
        rank_score = rank_score - (
            0.25 * ranked_df['cf_has_payment'].astype(float)
        )

    if 'dt_execution_state' in ranked_df.columns:
        completed = (
            ranked_df['dt_execution_state']
            .astype(str)
            .str.lower()
            .eq('completed')
            .astype(float)
        )
        rank_score = rank_score - (0.15 * completed)

    if (
        'cf_has_penalty' in ranked_df.columns
        and 'cf_has_payment' in ranked_df.columns
    ):
        penalty_without_payment = (
            ranked_df['cf_has_penalty'].astype(float).eq(1)
            & ranked_df['cf_has_payment'].astype(float).eq(0)
        ).astype(float)
        rank_score = rank_score + (0.35 * penalty_without_payment)

    if (
        'cf_has_appeal' in ranked_df.columns
        and 'cf_has_payment' in ranked_df.columns
    ):
        appeal_without_payment = (
            ranked_df['cf_has_appeal'].astype(float).eq(1)
            & ranked_df['cf_has_payment'].astype(float).eq(0)
        ).astype(float)
        rank_score = rank_score + (0.20 * appeal_without_payment)

    ranked_df['_baseline_fit_score'] = rank_score

    stable_df = (
        ranked_df
        .sort_values(
            by=['_baseline_fit_score', score_col, case_col],
            ascending=[True, True, True],
            kind='mergesort'
        )
        .head(keep_count)
    )

    return set(
        stable_df[case_col]
        .astype(str)
    )


def choose_iterative_baseline_threshold(
    scored_df,
    stable_case_ids,
    score_col='anomaly_score',
    case_col='case_id',
    quantile=DEFAULT_BASELINE_THRESHOLD_QUANTILE
):
    if scored_df.empty:
        return 0.0, f'{score_col}_iterative_baseline_empty'

    stable_scores = scored_df.loc[
        scored_df[case_col].astype(str).isin(stable_case_ids),
        score_col
    ].astype(float)

    if stable_scores.empty:
        stable_scores = scored_df[score_col].astype(float)

    threshold = float(
        stable_scores.quantile(quantile)
    )

    return (
        round(threshold, 4),
        f'{score_col}_iterative_baseline_p{int(quantile * 100)}'
    )


def apply_process_completion_rules(df, prediction_col='predicted_label'):
    predictions = df[prediction_col].copy()

    if (
        'cf_has_penalty' in df.columns
        and 'cf_has_payment' in df.columns
    ):
        penalty_without_payment = (
            df['cf_has_penalty'].astype(float).eq(1)
            & df['cf_has_payment'].astype(float).eq(0)
        )
        predictions.loc[penalty_without_payment] = 'deviant'

    if (
        'cf_has_appeal' in df.columns
        and 'cf_has_payment' in df.columns
    ):
        appeal_without_payment = (
            df['cf_has_appeal'].astype(float).eq(1)
            & df['cf_has_payment'].astype(float).eq(0)
        )
        predictions.loc[appeal_without_payment] = 'deviant'

    return predictions
