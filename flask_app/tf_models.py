"""
Model components for the Traffic Fines DT-IB anomaly detection pipeline.
"""

import json

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.metrics import roc_auc_score


def _safe_float(value, default=0.0):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default

    if pd.isna(number):
        return default

    return number


def _clip01(value):
    return float(np.clip(_safe_float(value), 0.0, 1.0))


class DigitalTwin:
    """Virtual process representation with adaptive numeric baselines."""

    def __init__(self, ewma_alpha=0.20):
        self.baseline = {}
        self.percentiles = {}
        self.ewma_thresholds = {}
        self.ewma_alpha = ewma_alpha
        self.case_states = {}

    def fit(self, df, numeric_cols):
        for col in numeric_cols:
            vals = pd.to_numeric(df[col], errors='coerce').dropna()

            if vals.empty:
                vals = pd.Series([0.0])

            q1 = vals.quantile(0.25)
            q3 = vals.quantile(0.75)
            iqr = q3 - q1
            std = vals.std()

            self.baseline[col] = {
                'mean': float(vals.mean()),
                'std': float(0.0 if pd.isna(std) else std),
                'median': float(vals.median()),
                'q1': float(q1),
                'q3': float(q3),
                'iqr': float(0.0 if pd.isna(iqr) else iqr),
                'p05': float(vals.quantile(0.05)),
                'p50': float(vals.quantile(0.50)),
                'p75': float(vals.quantile(0.75)),
                'p90': float(vals.quantile(0.90)),
                'p95': float(vals.quantile(0.95)),
            }

            self.percentiles[col] = np.percentile(
                vals,
                np.arange(0, 101, 5)
            )

            self.ewma_thresholds[col] = {
                'upper': float(vals.quantile(0.95)),
                'lower': float(vals.quantile(0.05)),
            }

        return self

    def seed_states(self, states):
        self.case_states = states

    def update_case_state(self, event):
        case_id = str(event.get('case_id', 'unknown'))
        activity = str(event.get('activity', 'unknown'))
        resource = str(event.get('resource', 'unknown'))
        timestamp = event.get('timestamp')

        state = self.case_states.setdefault(
            case_id,
            {
                'last_activity': None,
                'first_timestamp': timestamp,
                'last_timestamp': timestamp,
                'elapsed_time': 0.0,
                'resource_history': [],
                'activity_history': [],
                'execution_state': 'running',
            }
        )

        if state['first_timestamp'] is None:
            state['first_timestamp'] = timestamp

        if timestamp is not None and state['first_timestamp'] is not None:
            elapsed = timestamp - state['first_timestamp']
            state['elapsed_time'] = max(
                0.0,
                elapsed.total_seconds() / 3600
            )

        state['last_activity'] = activity
        state['last_timestamp'] = timestamp
        state['resource_history'].append(resource)
        state['activity_history'].append(activity)

        lowered = activity.lower()
        if any(token in lowered for token in ['payment', 'archive', 'close']):
            state['execution_state'] = 'completed'

        return state

    def update_ewma(self, row, numeric_cols):
        for col in numeric_cols:
            if col not in row.index:
                continue

            val = _safe_float(row[col])
            thresholds = self.ewma_thresholds.get(col)

            if thresholds is None:
                continue

            if val > thresholds['upper']:
                thresholds['upper'] = (
                    self.ewma_alpha * val
                    + (1 - self.ewma_alpha) * thresholds['upper']
                )

            if val < thresholds['lower']:
                thresholds['lower'] = (
                    self.ewma_alpha * val
                    + (1 - self.ewma_alpha) * thresholds['lower']
                )

    def get_z_score(self, row, col):
        b = self.baseline[col]
        std = b['std']

        if std == 0:
            return 0.0

        return abs((_safe_float(row[col]) - b['mean']) / std)

    def get_iqr_score(self, row, col):
        b = self.baseline[col]
        iqr = b['iqr']

        if iqr == 0:
            return 0.0

        value = _safe_float(row[col])

        return max(
            0.0,
            (value - b['q3']) / iqr,
            (b['q1'] - value) / iqr
        )


class DynamicDeclarativeConstraints:
    """Adaptive multi-perspective declarative constraints."""

    def __init__(
        self,
        z_threshold=2.5,
        iqr_multiplier=2.0,
        ewma_margin=0.10
    ):
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier
        self.ewma_margin = ewma_margin
        self.constraints = {}
        self.hard_rules = {}
        self.percentile_bounds = {}
        self.cf_profile = {}
        self.resource_profile = {}
        self.digital_twin = None

    def fit(
        self,
        digital_twin,
        numeric_cols,
        cf_profile=None,
        resource_profile=None
    ):
        self.digital_twin = digital_twin
        self.cf_profile = cf_profile or {}
        self.resource_profile = resource_profile or {}

        for col in numeric_cols:
            b = digital_twin.baseline[col]
            std = b['std']
            iqr = b['iqr']

            self.constraints[col] = {
                'upper_z': b['mean'] + self.z_threshold * std,
                'lower_z': b['mean'] - self.z_threshold * std,
                'upper_iqr': b['q3'] + self.iqr_multiplier * iqr,
                'lower_iqr': b['q1'] - self.iqr_multiplier * iqr,
            }

            percentiles = digital_twin.percentiles[col]

            self.percentile_bounds[col] = {
                'lower': float(percentiles[1]),
                'upper': float(percentiles[-2]),
            }

        self.hard_rules = {
            'cf_missing_steps': (0, 0, 'Control-Flow'),
            'cf_duplicate_steps': (0, 2, 'Control-Flow'),
            'cf_seq_violations': (0, 1, 'Control-Flow'),
            'res_unusual_activity_count': (0, 0, 'Resource'),
        }

        return self

    def _perspective_for_col(self, col):
        if col.startswith('cf_'):
            return 'Control-Flow'

        if col.startswith('temp_'):
            return 'Temporal'

        if col.startswith('res_'):
            return 'Resource'

        return 'Other'

    def evaluate(self, row):
        violations = []
        z_scores = {}
        perspective_hits = {
            'Control-Flow': 0.0,
            'Temporal': 0.0,
            'Resource': 0.0,
        }
        perspective_total = {
            'Control-Flow': 0,
            'Temporal': 0,
            'Resource': 0,
        }

        for col, constraint in self.constraints.items():
            if col not in row.index:
                continue

            val = _safe_float(row[col])
            perspective = self._perspective_for_col(col)

            if perspective in perspective_total:
                perspective_total[perspective] += 1

            center = (constraint['upper_z'] + constraint['lower_z']) / 2
            scale = max(
                1e-9,
                (constraint['upper_z'] - constraint['lower_z'])
                / max(1e-9, 2 * self.z_threshold)
            )
            z = abs((val - center) / scale)
            z_scores[f'{col}_z'] = z

            col_violations = []

            if val > constraint['upper_z'] or val < constraint['lower_z']:
                col_violations.append(f'DDC_Z:{col}')

            if val > constraint['upper_iqr'] or val < constraint['lower_iqr']:
                col_violations.append(f'DDC_IQR:{col}')

            p = self.percentile_bounds[col]
            if val > p['upper'] or val < p['lower']:
                col_violations.append(f'DDC_PCTL:{col}')

            ewma = self.digital_twin.ewma_thresholds.get(col, {})
            upper_ewma = ewma.get('upper')
            lower_ewma = ewma.get('lower')

            if upper_ewma is not None and val > upper_ewma * (1 + self.ewma_margin):
                col_violations.append(f'DDC_EWMA_HIGH:{col}')

            if lower_ewma is not None and val < lower_ewma * (1 - self.ewma_margin):
                col_violations.append(f'DDC_EWMA_LOW:{col}')

            if col_violations:
                violations.extend(col_violations)

                if perspective in perspective_hits:
                    severity = min(1.0, max(z / 3.0, len(col_violations) / 4.0))
                    perspective_hits[perspective] += severity

        for col, rule in self.hard_rules.items():
            if col not in row.index:
                continue

            lo, hi, perspective = rule
            val = _safe_float(row[col])

            if val < lo or val > hi:
                violations.append(f'DDC_HARD:{col}={val:.0f}')
                perspective_hits[perspective] += 1.0
                perspective_total[perspective] += 1

        if _safe_float(row.get('cf_wrong_order_ratio', 0.0)) > 0:
            violations.append('DDC_CF:wrong_order_transition')
            perspective_hits['Control-Flow'] += min(
                1.0,
                _safe_float(row.get('cf_wrong_order_ratio', 0.0))
            )
            perspective_total['Control-Flow'] += 1

        if _safe_float(row.get('res_unusual_activity_count', 0.0)) > 0:
            violations.append('DDC_RES:unusual_activity_resource')
            perspective_hits['Resource'] += min(
                1.0,
                _safe_float(row.get('res_unusual_activity_ratio', 0.0))
            )
            perspective_total['Resource'] += 1

        perspective_scores = {
            key: _clip01(perspective_hits[key] / max(perspective_total[key], 1))
            for key in perspective_hits
        }

        return violations, z_scores, perspective_scores


class MVARMiner:
    """Cross-perspective multi-view association rule miner."""

    def __init__(
        self,
        min_support=0.06,
        min_confidence=0.65,
        min_lift=1.0,
        max_rules=40
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
                0.50 * _safe_float(rule['confidence'])
                + 0.30 * min(_safe_float(rule['lift']) / 3.0, 1.0)
                + 0.20 * min(_safe_float(rule['support']) / max(self.min_support, 1e-9), 1.0)
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
                    0.50 * _safe_float(rule['confidence'])
                    + 0.30 * min(_safe_float(rule['lift']) / 3.0, 1.0)
                    + 0.20 * min(_safe_float(rule['support']) / max(self.min_support, 1e-9), 1.0)
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


class IntelligentBody:
    """Reasoning engine that fuses DDC, MV-ARM, z-score, and business evidence."""

    def __init__(self, digital_twin, ddc, mv_arm_miner):
        self.dt = digital_twin
        self.ddc = ddc
        self.mva = mv_arm_miner
        self.all_scores = []
        self.component_weights = None
        self.numeric_cols = []

    def calibrate_weights(self, df, numeric_cols):
        self.numeric_cols = numeric_cols

        if 'label' not in df.columns or df['label'].nunique() != 2:
            self.component_weights = None
            return

        calibration_df = df

        if len(df) > 5000:
            calibration_parts = []

            for _, group in df.groupby('label'):
                calibration_parts.append(
                    group.sample(
                        min(len(group), 2500),
                        random_state=42
                    )
                )

            calibration_df = pd.concat(
                calibration_parts,
                ignore_index=True
            )

        y_true = calibration_df['label'].eq('deviant').astype(int)
        component_rows = [
            self._component_scores(row)
            for _, row in calibration_df.iterrows()
        ]

        component_df = pd.DataFrame(component_rows)
        raw_weights = {}

        for col in ['ddc_score', 'z_score', 'arm_score', 'br_score']:
            try:
                if component_df[col].nunique() <= 1:
                    auc = 0.5
                else:
                    auc = roc_auc_score(y_true, component_df[col])
            except Exception:
                auc = 0.5

            raw_weights[col] = max(0.03, auc - 0.45)

        total = sum(raw_weights.values())

        if total <= 0:
            self.component_weights = None
            return

        self.component_weights = {
            key: value / total
            for key, value in raw_weights.items()
        }

    def _component_scores(self, row):
        ddc_viols, z_scores, perspective_scores = self.ddc.evaluate(row)

        total_possible_constraints = (
            len(self.ddc.constraints)
            + len(self.ddc.hard_rules)
            + 2
        )

        ddc_score = _clip01(
            len(ddc_viols) / max(total_possible_constraints, 1)
        )

        key_z_features = [
            'cf_seq_violations',
            'cf_missing_steps',
            'temp_total_hrs',
            'temp_max_step_hrs',
            'res_n_resources',
            'res_unusual_activity_count',
            'amount',
        ]

        key_z_features = [
            col for col in key_z_features
            if col in row.index and col in self.dt.baseline
        ]

        z_vals = np.array([
            self.dt.get_z_score(row, col)
            for col in key_z_features
        ])

        if len(z_vals) == 0:
            z_vals = np.array([0.0])

        z_composite = _clip01(float(np.mean(np.clip(z_vals / 3.0, 0, 1))))

        row_df = pd.DataFrame([row])
        if '_pre_arm_score' in row.index:
            arm_score = _clip01(row.get('_pre_arm_score', 0.0))
            triggered = []

            if arm_score > 0:
                triggered = [{
                    'view': 'Cross-View',
                    'rule': str(row.get('_pre_violated_arm_rules', '')),
                    'missing': 'expected consequent',
                    'confidence': 0.0,
                    'lift': 0.0,
                    'support': 0.0,
                    'severity': arm_score,
                }]
        else:
            triggered = self.mva.check_case(row_df)
            arm_score = self.mva.score_hits(triggered)

        business_rules = [
            _safe_float(row.get('cf_seq_violations', 0)) > 1,
            _safe_float(row.get('cf_missing_steps', 0)) > 0,
            (
                _safe_float(row.get('cf_has_appeal', 0)) == 1
                and _safe_float(row.get('cf_has_payment', 0)) == 0
            ),
            (
                _safe_float(row.get('cf_has_penalty', 0)) == 1
                and _safe_float(row.get('cf_has_payment', 0)) == 0
            ),
            (
                _safe_float(row.get('temp_total_hrs', 0))
                > self.dt.baseline['temp_total_hrs']['p90']
            ),
            _safe_float(row.get('res_unusual_activity_count', 0)) > 0,
        ]

        br_score = _clip01(sum(business_rules) / max(len(business_rules), 1))

        return {
            'ddc_score': ddc_score,
            'z_score': z_composite,
            'arm_score': arm_score,
            'br_score': br_score,
            'control_flow_score': perspective_scores['Control-Flow'],
            'temporal_score': perspective_scores['Temporal'],
            'resource_score': perspective_scores['Resource'],
            'ddc_violations': ddc_viols,
            'z_vals': z_vals,
            'z_features': key_z_features,
            'triggered': triggered,
        }

    def _fusion_weights(self, scores):
        if self.component_weights is not None:
            return self.component_weights

        evidence = {
            'ddc_score': scores['ddc_score'],
            'z_score': scores['z_score'],
            'arm_score': scores['arm_score'],
            'br_score': scores['br_score'],
        }

        total = sum(evidence.values())

        if total <= 1e-9:
            return {
                'ddc_score': 0.30,
                'z_score': 0.20,
                'arm_score': 0.30,
                'br_score': 0.20,
            }

        return {
            key: value / total
            for key, value in evidence.items()
        }

    def score_case(self, row):
        scores = self._component_scores(row)
        weights = self._fusion_weights(scores)

        composite = _clip01(
            scores['ddc_score'] * weights['ddc_score']
            + scores['z_score'] * weights['z_score']
            + scores['arm_score'] * weights['arm_score']
            + scores['br_score'] * weights['br_score']
        )

        self.all_scores.append(composite)

        if len(self.all_scores) >= 10:
            high_threshold = float(np.percentile(self.all_scores, 90))
            medium_threshold = float(np.percentile(self.all_scores, 70))
        else:
            current_scores = self.all_scores if self.all_scores else [composite]
            high_threshold = float(np.max(current_scores))
            medium_threshold = float(np.median(current_scores))

        if composite >= high_threshold:
            risk = 'High'
        elif composite >= medium_threshold:
            risk = 'Medium'
        else:
            risk = 'Low'

        anomaly_types = []

        if scores['control_flow_score'] > 0 or _safe_float(row.get('cf_missing_steps', 0)) > 0:
            anomaly_types.append('Control-Flow')

        if scores['temporal_score'] > 0 or scores['z_score'] > 0.35:
            anomaly_types.append('Temporal')

        if scores['resource_score'] > 0 or _safe_float(row.get('res_unusual_activity_count', 0)) > 0:
            anomaly_types.append('Resource')

        if not anomaly_types:
            anomaly_types.append('None')

        if self.numeric_cols:
            self.dt.update_ewma(row, self.numeric_cols)

        violated_constraints = scores['ddc_violations']
        violated_arm_rules = [
            hit['rule'] for hit in scores['triggered'][:5]
        ]

        return {
            'anomaly_score': round(composite, 4),
            'risk_level': risk,
            'anomaly_types': '|'.join(anomaly_types),
            'ddc_score': round(scores['ddc_score'], 4),
            'z_score': round(scores['z_score'], 4),
            'arm_score': round(scores['arm_score'], 4),
            'br_score': round(scores['br_score'], 4),
            'control_flow_score': round(scores['control_flow_score'], 4),
            'temporal_score': round(scores['temporal_score'], 4),
            'resource_score': round(scores['resource_score'], 4),
            'ddc_violations': len(violated_constraints),
            'arm_rules_hit': int(row.get('_pre_arm_rules_hit', len(scores['triggered']))),
            'high_threshold': round(high_threshold, 4),
            'medium_threshold': round(medium_threshold, 4),
            'adaptive_z_threshold': round(float(np.mean(scores['z_vals']) + np.std(scores['z_vals'])), 4),
            'fusion_weights': json.dumps({
                key.replace('_score', ''): round(value, 3)
                for key, value in weights.items()
            }),
            'violated_constraints': '; '.join(violated_constraints[:8]),
            'violated_arm_rules': str(row.get(
                '_pre_violated_arm_rules',
                '; '.join(violated_arm_rules)
            )),
            'explanation': self._explain(
                violated_constraints,
                scores['triggered'],
                anomaly_types,
                scores['z_vals'],
                scores['z_features'],
                row
            ),
        }

    def _explain(self, ddc_viols, triggered, atypes, z_vals, feat_names, row):
        parts = []

        if ddc_viols:
            parts.append(f"DDC violations: {'; '.join(ddc_viols[:3])}")

        top_z = sorted(
            zip(feat_names, z_vals),
            key=lambda x: -x[1]
        )[:2]

        if top_z and top_z[0][1] > 2:
            parts.append(
                f"High z-score: {top_z[0][0]} (z={top_z[0][1]:.2f})"
            )

        if triggered:
            first = triggered[0]
            parts.append(
                "MV-ARM violation: "
                f"{first['rule']} missing {first['missing']}"
            )

        if _safe_float(row.get('res_unusual_activity_count', 0)) > 0:
            parts.append(
                "Resource violation: unusual resource assignment for activity"
            )

        if len(atypes) > 1 and atypes != ['None']:
            parts.append(f"Cross-perspective evidence: {' + '.join(atypes)}")

        return ' | '.join(parts) if parts else 'No significant deviations detected'

    def score_all(self, df):
        results = []

        for _, row in df.iterrows():
            result = self.score_case(row)
            result['case_id'] = row['case_id']
            results.append(result)

        return pd.DataFrame(results)

    # =====================================================
    # CLEANED VERSION
    # =====================================================
    def score_all_fast(self, df):

        index = df.index

        ddc_counts = pd.Series(0.0, index=index)
        first_violation = pd.Series('', index=index, dtype=object)

        for col, constraint in self.ddc.constraints.items():

            if col not in df.columns:
                continue

            values = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

            masks = {
                f'DDC_Z:{col}':
                    (values > constraint['upper_z']) |
                    (values < constraint['lower_z']),

                f'DDC_IQR:{col}':
                    (values > constraint['upper_iqr']) |
                    (values < constraint['lower_iqr']),

                f'DDC_PCTL:{col}':
                    (values > self.ddc.percentile_bounds[col]['upper']) |
                    (values < self.ddc.percentile_bounds[col]['lower'])
            }

            ewma = self.dt.ewma_thresholds.get(col, {})

            if 'upper' in ewma:
                masks[f'DDC_EWMA_HIGH:{col}'] = (
                    values > ewma['upper'] * (1 + self.ddc.ewma_margin)
                )

            if 'lower' in ewma:
                masks[f'DDC_EWMA_LOW:{col}'] = (
                    values < ewma['lower'] * (1 - self.ddc.ewma_margin)
                )

            for label, mask in masks.items():

                mask = mask.fillna(False)

                ddc_counts += mask.astype(float)

                first_violation = first_violation.mask(
                    (first_violation == '') & mask,
                    label
                )

        for col, rule in self.ddc.hard_rules.items():

            if col not in df.columns:
                continue

            lo, hi, _ = rule

            values = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

            mask = (values < lo) | (values > hi)

            ddc_counts += mask.astype(float)

            first_violation = first_violation.mask(
                (first_violation == '') & mask,
                f'DDC_HARD:{col}'
            )

        total_constraints = (
            len(self.ddc.constraints) +
            len(self.ddc.hard_rules) + 2
        )

        ddc_score = np.clip(
            ddc_counts / max(total_constraints, 1),
            0,
            1
        )

        key_z_features = [
            'cf_seq_violations',
            'cf_missing_steps',
            'temp_total_hrs',
            'temp_max_step_hrs',
            'res_n_resources',
            'res_unusual_activity_count',
            'amount'
        ]

        z_parts = []

        for col in key_z_features:

            if col not in df.columns or col not in self.dt.baseline:
                continue

            b = self.dt.baseline[col]

            std = max(b['std'], 1e-9)

            values = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

            z_parts.append(
                ((values - b['mean']).abs() / std).clip(0, 3) / 3
            )

        z_score = (
            pd.concat(z_parts, axis=1).mean(axis=1)
            if z_parts else pd.Series(0.0, index=index)
        )

        if '_pre_arm_score' in df.columns:

            arm_score = pd.to_numeric(
                df['_pre_arm_score'],
                errors='coerce'
            ).fillna(0.0).clip(0, 1)

        else:

            arm_df = self.mva.score_dataframe(df)

            arm_score = arm_df['arm_score']

        br_rules = pd.concat([

            (df['cf_seq_violations'] > 1).astype(float),

            (df['cf_missing_steps'] > 0).astype(float),

            (
                (df['cf_has_appeal'] == 1) &
                (df['cf_has_payment'] == 0)
            ).astype(float),

            (
                (df['cf_has_penalty'] == 1) &
                (df['cf_has_payment'] == 0)
            ).astype(float),

            (
                df['temp_total_hrs'] >
                self.dt.baseline['temp_total_hrs']['p90']
            ).astype(float),

            (
                df['res_unusual_activity_count'] > 0
            ).astype(float)

        ], axis=1)

        br_score = br_rules.mean(axis=1)

        if self.component_weights is not None:

            w = self.component_weights

            composite = (
                ddc_score * w['ddc_score'] +
                z_score * w['z_score'] +
                arm_score * w['arm_score'] +
                br_score * w['br_score']
            )

        else:

            score_sum = (
                ddc_score +
                z_score +
                arm_score +
                br_score
            ).replace(0, np.nan)

            weights_df = pd.DataFrame({
                'ddc_score': ddc_score / score_sum,
                'z_score': z_score / score_sum,
                'arm_score': arm_score / score_sum,
                'br_score': br_score / score_sum,
            }).fillna({
                'ddc_score': 0.30,
                'z_score': 0.20,
                'arm_score': 0.30,
                'br_score': 0.20,
            })

            composite = (
                ddc_score * weights_df['ddc_score'] +
                z_score * weights_df['z_score'] +
                arm_score * weights_df['arm_score'] +
                br_score * weights_df['br_score']
            )

        composite = composite.clip(0, 1)

        high_threshold = float(np.percentile(composite, 90))
        medium_threshold = float(np.percentile(composite, 70))

        risk_level = np.where(
            composite >= high_threshold,
            'High',
            np.where(composite >= medium_threshold, 'Medium', 'Low')
        )

        anomaly_types = []
        explanations = []

        for idx in index:

            types = []

            if df.loc[idx, 'cf_missing_steps'] > 0:
                types.append('Control-Flow')

            if z_score.loc[idx] > 0.35:
                types.append('Temporal')

            if df.loc[idx, 'res_unusual_activity_count'] > 0:
                types.append('Resource')

            if not types:
                types.append('None')

            anomaly_types.append('|'.join(types))

            parts = []

            if first_violation.loc[idx]:
                parts.append(f"DDC violations: {first_violation.loc[idx]}")

            if z_score.loc[idx] > 0.67:
                parts.append("High z-score evidence")

            explanations.append(
                ' | '.join(parts)
                if parts else
                'No significant deviations detected'
            )

        return pd.DataFrame({

            'case_id': df['case_id'].values,
            'anomaly_score': composite.round(4),
            'risk_level': risk_level,
            'anomaly_types': anomaly_types,
            'ddc_score': ddc_score.round(4),
            'z_score': z_score.round(4),
            'arm_score': arm_score.round(4),
            'br_score': br_score.round(4),
            'explanation': explanations,

        })