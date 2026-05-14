"""
tf_models.py — Definisi semua class model Traffic Fines Anomaly Detection.
"""

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


class DigitalTwin:

    def __init__(self):
        self.baseline = {}
        self.percentiles = {}

    def fit(self, df, numeric_cols):

        for col in numeric_cols:

            vals = df[col].dropna()

            self.baseline[col] = {
                'mean': vals.mean(),
                'std': vals.std(),
                'median': vals.median(),
                'q1': vals.quantile(0.25),
                'q3': vals.quantile(0.75),
                'iqr': vals.quantile(0.75) - vals.quantile(0.25),
            }

            self.percentiles[col] = np.percentile(
                vals,
                np.arange(0, 101, 5)
            )

    def get_z_score(self, row, col):

        b = self.baseline[col]

        if b['std'] == 0:
            return 0.0

        return abs(
            (row[col] - b['mean']) / b['std']
        )

    def get_iqr_score(self, row, col):

        b = self.baseline[col]

        if b['iqr'] == 0:
            return 0.0

        return max(
            0,
            (row[col] - b['q3']) / b['iqr'],
            (b['q1'] - row[col]) / b['iqr']
        )


class DynamicDeclarativeConstraints:

    def __init__(
        self,
        z_threshold=2.5,
        iqr_multiplier=2.0
    ):

        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier

        self.constraints = {}
        self.hard_rules = {}
        self.percentile_bounds = {}

    def fit(self, digital_twin, numeric_cols):

        for col in numeric_cols:

            b = digital_twin.baseline[col]

            self.constraints[col] = {
                'upper_z': b['mean'] + self.z_threshold * b['std'],
                'lower_z': b['mean'] - self.z_threshold * b['std'],
                'upper_iqr': b['q3'] + self.iqr_multiplier * b['iqr'],
                'lower_iqr': b['q1'] - self.iqr_multiplier * b['iqr'],
            }

            percentiles = digital_twin.percentiles[col]

            self.percentile_bounds[col] = {
                'lower': percentiles[1],
                'upper': percentiles[-2],
            }

        self.hard_rules = {
            'cf_seq_violations': (0, 2),
            'cf_missing_steps': (0, 0),
            'cf_duplicate_steps': (0, 3),
        }

    def evaluate(self, row):

        violations = []
        scores = {}

        for col, c in self.constraints.items():

            val = row[col]

            z = abs(
                (
                    val
                    - (c['upper_z'] + c['lower_z']) / 2
                )
                /
                max(
                    1e-9,
                    (
                        c['upper_z']
                        - c['lower_z']
                    )
                    /
                    (
                        2
                        * self.z_threshold
                    )
                )
            )

            scores[f'{col}_z'] = z

            if val > c['upper_z'] or val < c['lower_z']:
                violations.append(f'DDC_Z:{col}')

            if val > c['upper_iqr'] or val < c['lower_iqr']:
                violations.append(f'DDC_IQR:{col}')

            p = self.percentile_bounds[col]

            if val > p['upper'] or val < p['lower']:
                violations.append(f'DDC_PCTL:{col}')

        for col, (lo, hi) in self.hard_rules.items():

            if (
                col in row.index
                and (
                    row[col] < lo
                    or row[col] > hi
                )
            ):

                violations.append(
                    f'DDC_HARD:{col}={row[col]:.0f}'
                )

        return violations, scores


class MVARMiner:

    def __init__(
        self,
        min_support=0.05,
        min_confidence=0.6,
        min_lift=1.2
    ):

        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift

        self.rules = {
            'cf': None,
            'temporal': None,
            'resource': None
        }

        self._items_func = self._discretize

    def _discretize(self, df):

        items = pd.DataFrame(index=df.index)

        # ===== CONTROL FLOW =====
        items['CF_seq_violation'] = (
            df['cf_seq_violations'] > 0
        )

        items['CF_missing_step'] = (
            df['cf_missing_steps'] > 0
        )

        items['CF_duplicate'] = (
            df['cf_duplicate_steps'] > 0
        )

        items['CF_has_appeal'] = (
            df['cf_has_appeal'] == 1
        )

        items['CF_has_penalty'] = (
            df['cf_has_penalty'] == 1
        )

        items['CF_no_payment'] = (
            df['cf_has_payment'] == 0
        )

        items['CF_too_many_events'] = (
            df['cf_n_events']
            >
            df['cf_n_events'].quantile(0.90)
        )

        # ===== TEMPORAL =====
        med_hrs = df['temp_total_hrs'].median()
        std_hrs = df['temp_total_hrs'].std()

        items['TEMP_very_fast'] = (
            df['temp_total_hrs']
            < (med_hrs - std_hrs)
        )

        items['TEMP_very_slow'] = (
            df['temp_total_hrs']
            > (med_hrs + std_hrs)
        )

        items['TEMP_high_variance'] = (
            df['temp_std_step_hrs']
            >
            df['temp_std_step_hrs'].quantile(0.75)
        )

        items['TEMP_long_max_step'] = (
            df['temp_max_step_hrs']
            >
            df['temp_max_step_hrs'].quantile(0.90)
        )

        # ===== RESOURCE =====
        items['RES_multi_resource'] = (
            df['res_n_resources'] > 1
        )

        items['RES_too_many_resources'] = (
            df['res_n_resources']
            >
            df['res_n_resources'].quantile(0.90)
        )

        items['RES_rpa_involved'] = (
            df['res_rpa_flag'] == 1
        )

        # ===== FINANCIAL =====
        items['RES_high_expense'] = (
            df['expense']
            >
            df['expense'].quantile(0.90)
        )

        items['RES_high_amount'] = (
            df['amount']
            >
            df['amount'].quantile(0.90)
        )

        return items

    def fit(self, train_df):

        items = self._discretize(train_df)

        for view, cols in [

            (
                'cf',
                [
                    c for c in items.columns
                    if c.startswith('CF_')
                ]
            ),

            (
                'temporal',
                [
                    c for c in items.columns
                    if c.startswith('TEMP_')
                ]
            ),

            (
                'resource',
                [
                    c for c in items.columns
                    if c.startswith('RES_')
                ]
            ),
        ]:

            sub = items[cols].astype(bool)

            try:

                freq = apriori(
                    sub,
                    min_support=self.min_support,
                    use_colnames=True
                )

                if len(freq) > 0:

                    rules = association_rules(
                        freq,
                        metric='confidence',
                        min_threshold=self.min_confidence
                    )

                    rules = rules[
                        rules['lift']
                        >= self.min_lift
                    ].copy()

                    self.rules[view] = rules

                else:
                    self.rules[view] = None

            except Exception:
                self.rules[view] = None

        self._items_func = self._discretize

        return self

    def check_case(self, row_df):

        items = self._items_func(row_df)

        triggered = []

        for view, rules in self.rules.items():

            if (
                rules is None
                or len(rules) == 0
            ):
                continue

            for _, rule in rules.iterrows():

                antecedent = set(
                    rule['antecedents']
                )

                if all(
                    items.get(
                        a,
                        pd.Series([False])
                    ).values[0]
                    for a in antecedent
                ):

                    triggered.append({
                        'view': view,
                        'rule': f"{set(rule['antecedents'])} => {set(rule['consequents'])}",
                        'confidence': round(rule['confidence'], 3),
                        'lift': round(rule['lift'], 3),
                        'support': round(rule['support'], 3),
                    })

        return triggered


class IntelligentBody:

    def __init__(
        self,
        digital_twin,
        ddc,
        mv_arm_miner
    ):

        self.dt = digital_twin
        self.ddc = ddc
        self.mva = mv_arm_miner

        # adaptive score distribution
        self.all_scores = []

    def score_case(self, row):

        ddc_viols, z_scores = self.ddc.evaluate(row)

        # =====================================================
        # DDC SCORE
        # =====================================================
        ddc_score = min(
            1.0,
            len(ddc_viols) / 6.0
        )

        # =====================================================
        # Z-SCORE COMPOSITE
        # =====================================================
        key_z_features = [
            'cf_seq_violations',
            'temp_total_hrs',
            'temp_max_step_hrs',
            'amount',
            'res_n_resources',
        ]

        z_vals = [
            self.dt.get_z_score(row, c)
            for c in key_z_features
        ]

        z_composite = min(
            1.0,
            np.mean(z_vals) / 3.0
        )

        # =====================================================
        # MV-ARM SCORE
        # =====================================================
        row_df = pd.DataFrame([row])

        triggered = self.mva.check_case(row_df)

        if len(triggered) > 0:

            arm_score = np.mean([
                r['confidence'] * r['lift']
                for r in triggered
            ])

        else:
            arm_score = 0.0

        arm_score = min(1.0, arm_score)

        # =====================================================
        # BUSINESS RULE SCORE
        # =====================================================
        rules = [

            # abnormal sequence
            row['cf_seq_violations'] > 1,

            # missing steps
            row['cf_missing_steps'] > 0,

            # appeal without payment
            (
                row['cf_has_appeal'] == 1
                and row['cf_has_payment'] == 0
            ),

            # penalty escalation
            (
                row['cf_has_penalty'] == 1
                and row['cf_has_payment'] == 0
            ),

            # abnormal duration
            (
                row['temp_total_hrs']
                >
                self.dt.baseline[
                    'temp_total_hrs'
                ]['q3']
            )
        ]

        total_rules = len(rules)

        triggered_rules = sum(rules)

        br_score = (
            triggered_rules
            /
            max(total_rules, 1)
        )

        br_score = min(1.0, br_score)

        # =====================================================
        # DYNAMIC WEIGHTED FUSION
        # =====================================================
        scores = np.array([
            ddc_score,
            z_composite,
            arm_score,
            br_score
        ])

        weights = scores / (
            scores.sum() + 1e-9
        )

        composite = np.sum(
            scores * weights
        )

        composite = float(
            min(1.0, composite)
        )

        # simpan distribusi score
        self.all_scores.append(composite)

        # =====================================================
        # ADAPTIVE RISK THRESHOLD
        # =====================================================
        if len(self.all_scores) >= 10:

            high_threshold = np.percentile(
                self.all_scores,
                90
            )

            medium_threshold = np.percentile(
                self.all_scores,
                70
            )

        else:

            # fallback awal
            high_threshold = 0.75
            medium_threshold = 0.50

        if composite >= high_threshold:
            risk = 'High'

        elif composite >= medium_threshold:
            risk = 'Medium'

        else:
            risk = 'Low'

        # =====================================================
        # ANOMALY TYPE IDENTIFICATION
        # =====================================================
        anomaly_types = []

        # control-flow anomaly
        if (
            row['cf_seq_violations'] > 0
            or row['cf_missing_steps'] > 0
        ):

            anomaly_types.append(
                'Control-Flow'
            )

        # temporal anomaly
        if (
            z_vals[1] > 2.5
            or z_vals[2] > 2.5
        ):

            anomaly_types.append(
                'Temporal'
            )

        # resource anomaly
        resource_threshold = (
            self.dt.baseline[
                'res_n_resources'
            ]['q3']
        )

        if (
            row['res_n_resources']
            >
            resource_threshold
        ):

            anomaly_types.append(
                'Resource'
            )

        # automation / bot anomaly
        if row['res_rpa_flag'] == 1:

            anomaly_types.append(
                'Resource'
            )

        if not anomaly_types:
            anomaly_types.append('None')

        return {

            'anomaly_score': round(
                composite,
                4
            ),

            'risk_level': risk,

            'anomaly_types': '|'.join(
                anomaly_types
            ),

            'ddc_score': round(
                ddc_score,
                4
            ),

            'z_score': round(
                z_composite,
                4
            ),

            'arm_score': round(
                arm_score,
                4
            ),

            'br_score': round(
                br_score,
                4
            ),

            'ddc_violations': len(
                ddc_viols
            ),

            'arm_rules_hit': len(
                triggered
            ),

            'high_threshold': round(
                high_threshold,
                4
            ),

            'medium_threshold': round(
                medium_threshold,
                4
            ),

            'explanation': self._explain(
                ddc_viols,
                triggered,
                anomaly_types,
                z_vals,
                key_z_features
            ),
        }

    def _explain(
        self,
        ddc_viols,
        triggered,
        atypes,
        z_vals,
        feat_names
    ):

        parts = []

        if ddc_viols:

            parts.append(
                f"DDC violations: {'; '.join(ddc_viols[:3])}"
            )

        top_z = sorted(
            zip(feat_names, z_vals),
            key=lambda x: -x[1]
        )[:2]

        if top_z[0][1] > 2:

            parts.append(
                f"High z-score: {top_z[0][0]} "
                f"(z={top_z[0][1]:.2f})"
            )

        if triggered:

            parts.append(
                f"MV-ARM rule triggered: "
                f"{triggered[0]['view']} view"
            )

        return (
            ' | '.join(parts)
            if parts
            else
            'No significant deviations detected'
        )

    def score_all(self, df):

        results = []

        for _, row in df.iterrows():

            r = self.score_case(row)

            r['case_id'] = row['case_id']

            results.append(r)

        return pd.DataFrame(results)