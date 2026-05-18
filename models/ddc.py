import numpy as np
import pandas as pd

from utils.helpers import (
    _safe_float,
    _clip01
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

