import numpy as np
import pandas as pd

from utils.helpers import (
    _safe_float,
    _clip01
)

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

