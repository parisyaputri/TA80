import json

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

from utils.helpers import (
    _safe_float,
    _clip01
)

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