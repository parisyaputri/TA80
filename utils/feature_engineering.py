import numpy as np
import pandas as pd

from configs.model_config import ProcessConfig
from models.digital_twin import DigitalTwin

def learn_control_flow_profile(grouped, min_edge_support=ProcessConfig.MIN_EDGE_SUPPORT):
    edge_counts = {}
    activity_case_counts = {}
    activity_repetitions = {}
    total_cases = 0

    for _, group in grouped:
        total_cases += 1
        activities = list(group['_activity_norm'])

        for activity in set(activities):
            activity_case_counts[activity] = activity_case_counts.get(activity, 0) + 1

        counts = {}

        for activity in activities:
            counts[activity] = counts.get(activity, 0) + 1

        for activity, count in counts.items():
            activity_repetitions.setdefault(activity, []).append(count)

        for current_activity, next_activity in zip(activities, activities[1:]):
            edge = (current_activity, next_activity)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    min_edge_count = max(1, int(np.ceil(total_cases * min_edge_support)))

    frequent_edges = {
        edge
        for edge, count in edge_counts.items()
        if count >= min_edge_count
    }

    required_activities = {
        activity
        for activity, count in activity_case_counts.items()
        if count / max(total_cases, 1) >= ProcessConfig.REQUIRED_ACTIVITY_RATIO
    }

    max_repetitions = {
        activity: max(1, int(np.ceil(np.percentile(counts, ProcessConfig.MAX_REPETITION_PERCENTILE))))
        for activity, counts in activity_repetitions.items()
    }

    return {
        'frequent_edges': frequent_edges,
        'required_activities': required_activities,
        'max_repetitions': max_repetitions,
        'edge_counts': edge_counts,
        'total_cases': total_cases,
    }


def learn_resource_profile(profile_df):
    activity_resources = {}
    resource_counts = {}
    total_events = len(profile_df)

    for _, event in profile_df.iterrows():
        activity = event['_activity_norm']
        resource = event['_resource_norm']

        activity_resources.setdefault(activity, {})
        activity_resources[activity][resource] = (
            activity_resources[activity].get(resource, 0) + 1
        )
        resource_counts[resource] = resource_counts.get(resource, 0) + 1

    allowed_by_activity = {}

    for activity, counts in activity_resources.items():
        total = sum(counts.values())
        min_count = max(1, int(np.ceil(total * ProcessConfig.ALLOWED_RESOURCE_THRESHOLD)))
        allowed_by_activity[activity] = {
            resource
            for resource, count in counts.items()
            if count >= min_count
        }

    resource_frequency_share = {
        resource: count / max(total_events, 1)
        for resource, count in resource_counts.items()
    }

    return {
        'allowed_by_activity': allowed_by_activity,
        'resource_frequency_share': resource_frequency_share,
    }


def replay_event_states(cleaned_df):
    digital_twin = DigitalTwin()

    for _, event in cleaned_df.sort_values('_parsed_timestamp').iterrows():
        timestamp = event['_parsed_timestamp']

        if pd.isna(timestamp):
            timestamp = None

        digital_twin.update_case_state({
            'case_id': event['_case_id_norm'],
            'activity': event['_activity_norm'],
            'resource': event['_resource_norm'],
            'timestamp': timestamp,
        })

    return digital_twin.case_states

def build_case_features(cleaned_df, cf_profile, resource_profile, case_states, label_col):
    rows = []
    allowed_by_activity = resource_profile['allowed_by_activity']
    resource_frequency = resource_profile['resource_frequency_share']

    for case_id, group in cleaned_df.groupby('_case_id_norm', sort=False):
        group = group.sort_values('_parsed_timestamp')
        activities = list(group['_activity_norm'])
        resources = list(group['_resource_norm'])
        unique_resources = list(set(resources))

        activity_counts = {}

        for activity in activities:
            activity_counts[activity] = activity_counts.get(activity, 0) + 1

        frequent_edges = cf_profile['frequent_edges']
        edges = list(zip(activities, activities[1:]))

        seq_violations = sum(
            1
            for edge in edges
            if edge not in frequent_edges
        )

        wrong_order_ratio = (
            seq_violations / max(len(edges), 1)
            if edges else 0.0
        )

        missing_steps = len(
            cf_profile['required_activities'] - set(activities)
        )

        duplicate_steps = sum(
            max(
                0,
                count - cf_profile['max_repetitions'].get(activity, 1)
            )
            for activity, count in activity_counts.items()
        )

        timestamps = group['_parsed_timestamp'].dropna()
        total_hrs = 0.0
        max_step_hrs = 0.0
        std_step_hrs = 0.0

        if len(timestamps) > 1:
            total_hrs = (
                timestamps.max() - timestamps.min()
            ).total_seconds() / 3600

            diffs = timestamps.diff().dt.total_seconds().dropna() / 3600

            if len(diffs) > 0:
                max_step_hrs = float(diffs.max())
                std_step_hrs = float(0.0 if pd.isna(diffs.std()) else diffs.std())

        resource_frequency_in_case = {}

        for resource in resources:
            resource_frequency_in_case[resource] = (
                resource_frequency_in_case.get(resource, 0) + 1
            )

        max_resource_usage = (
            max(resource_frequency_in_case.values())
            if resource_frequency_in_case else 0
        )

        unusual_resource_events = 0

        for activity, resource in zip(activities, resources):
            allowed = allowed_by_activity.get(activity)

            if allowed is not None and resource not in allowed:
                unusual_resource_events += 1

        resource_rarity = 0.0

        if resources:
            resource_rarity = float(np.mean([
                1.0 - resource_frequency.get(resource, 0.0)
                for resource in resources
            ]))

        state = case_states.get(str(case_id), {})
        true_label = 'unknown'

        if label_col is not None:
            labels = group[label_col].astype(str).str.lower()
            true_label = 'deviant' if any(labels == 'deviant') else 'regular'

        res_n_resources = len(unique_resources)
        event_count = len(group)

        row = {
            'case_id': str(case_id),
            'label': true_label,
            'cf_n_events': event_count,
            'cf_seq_violations': seq_violations,
            'cf_missing_steps': missing_steps,
            'cf_duplicate_steps': duplicate_steps,
            'cf_wrong_order_ratio': wrong_order_ratio,
            'cf_has_appeal': int(any(any(kw in activity for kw in ProcessConfig.APPEAL_KEYWORDS) for activity in activities)),
            'cf_has_penalty': int(any(any(kw in activity for kw in ProcessConfig.PENALTY_KEYWORDS) for activity in activities)),
            'cf_has_payment': int(any(any(kw in activity for kw in ProcessConfig.PAYMENT_KEYWORDS) for activity in activities)),
            'temp_total_hrs': total_hrs,
            'temp_max_step_hrs': max_step_hrs,
            'temp_std_step_hrs': std_step_hrs,
            'res_n_resources': res_n_resources,
            'res_single_resource': int(res_n_resources == 1),
            'res_many_resources': res_n_resources,
            'res_dominant_resource_ratio': (
                max_resource_usage / max(len(resources), 1)
            ),
            'res_rpa_flag': int(any(
                any(keyword in resource for keyword in ProcessConfig.RPA_KEYWORDS)
                for resource in resources
            )),
            'res_unusual_activity_count': unusual_resource_events,
            'res_unusual_activity_ratio': (
                unusual_resource_events / max(event_count, 1)
            ),
            'res_workload_share': (
                max_resource_usage / max(event_count, 1)
            ),
            'res_resource_rarity': resource_rarity,
            'event_count': event_count,
            'dt_last_activity': state.get('last_activity', ''),
            'dt_execution_state': state.get('execution_state', 'running'),
            'amount': float(group['_amount_numeric'].mean()),
            'expense': float(group['_expense_numeric'].mean()),
        }

        rows.append(row)

    return pd.DataFrame(rows).fillna(0)
