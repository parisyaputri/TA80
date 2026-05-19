# utils/statistical_tests.py

import numpy as np

from scipy.stats import wilcoxon

from statsmodels.stats.multitest import multipletests


def run_wilcoxon_test(
    proposed_scores,
    baseline_scores
):

    stat, p_value = wilcoxon(
        proposed_scores,
        baseline_scores
    )

    return stat, p_value


def compute_cohens_d(
    proposed_scores,
    baseline_scores
):

    proposed_scores = np.array(proposed_scores)
    baseline_scores = np.array(baseline_scores)

    pooled_std = np.sqrt(
        (
            proposed_scores.std() ** 2 +
            baseline_scores.std() ** 2
        ) / 2
    )

    d = (
        proposed_scores.mean() -
        baseline_scores.mean()
    ) / pooled_std

    return d


def apply_bonferroni(p_values):

    corrected = multipletests(
        p_values,
        method='bonferroni'
    )

    return corrected[1]