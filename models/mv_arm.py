import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='mlxtend')

from configs.model_config import ARMConfig

from mlxtend.frequent_patterns import (
    apriori,
    association_rules
)

from utils.helpers import (
    _safe_float,
    _clip01
)

class MVARMiner:
    """Cross-perspective multi-view association rule miner."""

    def __init__(
        self,
        min_support=ARMConfig.MIN_SUPPORT,
        min_confidence=ARMConfig.MIN_CONFIDENCE,
        min_lift=ARMConfig.MIN_LIFT,
        max_rules=ARMConfig.MAX_RULES
    ):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.max_rules = max_rules
        self.rules = None
        self.single_view_rules = {
            'cf': None,
            'temporal': None,
            'resource': None
        }
        self.thresholds = {}

    def _threshold(self, df, col, q, default=0.0):
        if col not in df.columns:
            return default

        vals = pd.to_numeric(df[col], errors='coerce').dropna()

        if vals.empty:
            return default

        return float(vals.quantile(q))

    def _discretize(self, df):
        items = pd.DataFrame(index=df.index)
        t = self.thresholds

        items['CF_seq_violation'] = df['cf_seq_violations'] > 0
        items['CF_missing_step'] = df['cf_missing_steps'] > 0
        items['CF_duplicate'] = df['cf_duplicate_steps'] > 0
        items['CF_has_appeal'] = df['cf_has_appeal'] == 1
        items['CF_has_penalty'] = df['cf_has_penalty'] == 1
        items['CF_has_payment'] = df['cf_has_payment'] == 1
        items['CF_no_payment'] = df['cf_has_payment'] == 0
        items['CF_wrong_order'] = df.get('cf_wrong_order_ratio', 0) > 0
        items['CF_many_events'] = (
            df['cf_n_events'] > t.get('cf_n_events_q90', 0)
        )

        items['TEMP_slow'] = (
            df['temp_total_hrs'] > t.get('temp_total_q90', 0)
        )
        items['TEMP_very_slow'] = (
            df['temp_total_hrs'] > t.get('temp_total_q95', 0)
        )
        items['TEMP_long_step'] = (
            df['temp_max_step_hrs'] > t.get('temp_max_step_q90', 0)
        )
        items['TEMP_variable'] = (
            df['temp_std_step_hrs'] > t.get('temp_std_step_q75', 0)
        )
        items['TEMP_fast'] = (
            df['temp_total_hrs'] < t.get('temp_total_q10', 0)
        )

        items['RES_multi_resource'] = df['res_n_resources'] > 1
        items['RES_many_resources'] = (
            df['res_n_resources'] > t.get('res_n_resources_q90', 0)
        )
        items['RES_unusual_activity_resource'] = (
            df.get('res_unusual_activity_count', 0) > 0
        )
        items['RES_low_dominance'] = (
            df['res_dominant_resource_ratio']
            < t.get('res_dominance_q10', 0)
        )
        items['RES_rpa_involved'] = df['res_rpa_flag'] == 1
        items['RES_high_workload'] = (
            df.get('res_workload_share', 0) > t.get('res_workload_q90', 0)
        )

        items['FIN_high_amount'] = df['amount'] > t.get('amount_q90', 0)
        items['FIN_high_expense'] = df['expense'] > t.get('expense_q90', 0)

        return items.astype(bool)

    def fit(self, train_df):
        self.thresholds = {
            'cf_n_events_q90': self._threshold(train_df, 'cf_n_events', 0.90),
            'temp_total_q10': self._threshold(train_df, 'temp_total_hrs', 0.10),
            'temp_total_q90': self._threshold(train_df, 'temp_total_hrs', 0.90),
            'temp_total_q95': self._threshold(train_df, 'temp_total_hrs', 0.95),
            'temp_max_step_q90': self._threshold(train_df, 'temp_max_step_hrs', 0.90),
            'temp_std_step_q75': self._threshold(train_df, 'temp_std_step_hrs', 0.75),
            'res_n_resources_q90': self._threshold(train_df, 'res_n_resources', 0.90),
            'res_dominance_q10': self._threshold(train_df, 'res_dominant_resource_ratio', 0.10),
            'res_workload_q90': self._threshold(train_df, 'res_workload_share', 0.90),
            'amount_q90': self._threshold(train_df, 'amount', 0.90),
            'expense_q90': self._threshold(train_df, 'expense', 0.90),
        }

        items = self._discretize(train_df)
        self.rules = self._mine_rules(items)

        view_prefixes = {
            'cf': ('CF_',),
            'temporal': ('TEMP_',),
            'resource': ('RES_',),
        }

        for view, prefixes in view_prefixes.items():
            cols = [
                col for col in items.columns
                if col.startswith(prefixes)
            ]
            self.single_view_rules[view] = self._mine_rules(items[cols])

        return self

    def _mine_rules(self, items):
        if items.empty or items.shape[1] == 0:
            return None

        try:
            freq = apriori(
                items.astype(bool),
                min_support=self.min_support,
                use_colnames=True,
                max_len=3
            )

            if freq.empty:
                return None

            rules = association_rules(
                freq,
                metric='confidence',
                min_threshold=self.min_confidence
            )

            if rules.empty:
                return None

            rules = rules[
                (rules['lift'] >= self.min_lift)
                & (rules['consequents'].apply(len) == 1)
            ].copy()

            if rules.empty:
                return None

            rules['antecedent_views'] = rules['antecedents'].apply(self._views_for_items)
            rules['consequent_views'] = rules['consequents'].apply(self._views_for_items)
            rules['is_cross_view'] = rules.apply(
                lambda r: len(r['antecedent_views'] | r['consequent_views']) > 1,
                axis=1
            )

            rules = rules.sort_values(
                by=['is_cross_view', 'confidence', 'lift', 'support'],
                ascending=False
            )

            return rules.head(self.max_rules)

        except Exception:
            return None

    def _views_for_items(self, items):
        views = set()

        for item in items:
            if item.startswith('CF_'):
                views.add('Control-Flow')
            elif item.startswith('TEMP_'):
                views.add('Temporal')
            elif item.startswith('RES_'):
                views.add('Resource')
            elif item.startswith('FIN_'):
                views.add('Financial')

        return views

    def _check_rules(self, row_df, rules, require_cross_view=False):
        if rules is None or len(rules) == 0:
            return []

        items = self._discretize(row_df)
        triggered = []

        for _, rule in rules.iterrows():
            if require_cross_view and not bool(rule.get('is_cross_view', False)):
                continue

            antecedent = set(rule['antecedents'])
            consequent = set(rule['consequents'])

            antecedent_present = all(
                bool(items.get(item, pd.Series([False])).values[0])
                for item in antecedent
            )

            if not antecedent_present:
                continue

            missing_consequents = [
                item for item in consequent
                if not bool(items.get(item, pd.Series([False])).values[0])
            ]

            if not missing_consequents:
                continue

            views = self._views_for_items(antecedent | consequent)
            severity = (
                ARMConfig.CONFIDENCE_WEIGHT * _safe_float(rule['confidence'])
                + ARMConfig.LIFT_WEIGHT * min(_safe_float(rule['lift']) / ARMConfig.LIFT_NORM_DIVISOR, 1.0)
                + ARMConfig.SUPPORT_WEIGHT * min(_safe_float(rule['support']) / max(self.min_support, 1e-9), 1.0)
            )

            triggered.append({
                'view': '|'.join(sorted(views)),
                'rule': self._format_rule(antecedent, consequent),
                'missing': '|'.join(missing_consequents),
                'confidence': round(_safe_float(rule['confidence']), 3),
                'lift': round(_safe_float(rule['lift']), 3),
                'support': round(_safe_float(rule['support']), 3),
                'severity': round(_clip01(severity), 4),
            })

        return triggered

    def _format_rule(self, antecedent, consequent):
        left = ' + '.join(sorted(antecedent))
        right = ' + '.join(sorted(consequent))
        return f'{left} => {right}'

    def check_case(self, row_df):
        cross_view_hits = self._check_rules(
            row_df,
            self.rules,
            require_cross_view=True
        )

        if cross_view_hits:
            return cross_view_hits

        return self._check_rules(
            row_df,
            self.rules,
            require_cross_view=False
        )

    def check_single_view_case(self, row_df):
        hits = []

        for view, rules in self.single_view_rules.items():
            for hit in self._check_rules(row_df, rules, require_cross_view=False):
                hit['view'] = view
                hits.append(hit)

        return hits

    def score_dataframe(self, df, single_view=False):
        if single_view:
            rule_sets = [
                rules
                for rules in self.single_view_rules.values()
                if rules is not None and len(rules) > 0
            ]
        else:
            rule_sets = [
                self.rules
            ] if self.rules is not None and len(self.rules) > 0 else []

        if not rule_sets:
            return pd.DataFrame({
                'arm_score': np.zeros(len(df)),
                'arm_rules_hit': np.zeros(len(df), dtype=int),
                'violated_arm_rules': [''] * len(df),
            }, index=df.index)

        items = self._discretize(df)
        score_sum = np.zeros(len(df), dtype=float)
        hit_count = np.zeros(len(df), dtype=int)
        descriptions = [''] * len(df)
        require_cross = False

        if not single_view:
            require_cross = any(
                bool(rule.get('is_cross_view', False))
                for rules in rule_sets
                for _, rule in rules.iterrows()
            )

        for rules in rule_sets:
            for _, rule in rules.iterrows():
                if require_cross and not bool(rule.get('is_cross_view', False)):
                    continue

                antecedent = list(rule['antecedents'])
                consequent = list(rule['consequents'])

                if not antecedent or not consequent:
                    continue

                if any(item not in items.columns for item in antecedent + consequent):
                    continue

                antecedent_mask = items[antecedent].all(axis=1)
                missing_mask = ~items[consequent].all(axis=1)
                mask = (antecedent_mask & missing_mask).to_numpy()

                if not mask.any():
                    continue

                severity = (
                    ARMConfig.CONFIDENCE_WEIGHT * _safe_float(rule['confidence'])
                    + ARMConfig.LIFT_WEIGHT * min(_safe_float(rule['lift']) / ARMConfig.LIFT_NORM_DIVISOR, 1.0)
                    + ARMConfig.SUPPORT_WEIGHT * min(_safe_float(rule['support']) / max(self.min_support, 1e-9), 1.0)
                )
                severity = _clip01(severity)

                score_sum[mask] += severity
                hit_count[mask] += 1

                rule_text = self._format_rule(
                    set(rule['antecedents']),
                    set(rule['consequents'])
                )

                for idx in np.flatnonzero(mask):
                    if descriptions[idx] == '':
                        descriptions[idx] = rule_text

        scores = np.divide(
            score_sum,
            np.maximum(hit_count, 1),
            out=np.zeros_like(score_sum),
            where=hit_count > 0
        )

        return pd.DataFrame({
            'arm_score': np.clip(scores, 0, 1),
            'arm_rules_hit': hit_count,
            'violated_arm_rules': descriptions,
        }, index=df.index)

    def score_hits(self, hits):
        if not hits:
            return 0.0

        return _clip01(np.mean([_safe_float(hit.get('severity', 0.0)) for hit in hits]))

