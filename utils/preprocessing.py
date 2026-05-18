import pandas as pd

def safe_resource_value(value):
    if pd.isna(value):
        return 'unknown'

    text = str(value).strip().lower()
    return text if text else 'unknown'


def _normalise_name(value):
    if pd.isna(value):
        return 'unknown'

    return str(value).strip().lower()


def detect_event_log_columns(df):
    case_col = None
    activity_col = None
    timestamp_col = None
    resource_col = None
    label_col = None

    for col in df.columns:
        col_lower = col.lower()

        if label_col is None and 'label' in col_lower:
            label_col = col

        if (
            case_col is None
            and 'case' in col_lower
            and (
                'id' in col_lower
                or 'concept:name' in col_lower
            )
        ):
            case_col = col

        if (
            activity_col is None
            and (
                col_lower in ['activity', 'concept:name']
                or 'activity' in col_lower
            )
        ):
            activity_col = col

        if (
            resource_col is None
            and any(keyword in col_lower for keyword in [
                'resource',
                'org:resource',
                'user',
                'employee',
                'staff',
                'worker',
                'officer',
                'agent',
                'actor'
            ])
        ):
            resource_col = col

    timestamp_candidates = []

    for col in df.columns:
        col_lower = col.lower()

        if not any(keyword in col_lower for keyword in [
            'timestamp',
            'complete timestamp',
            'time:timestamp',
            'date'
        ]):
            continue

        parsed = pd.to_datetime(df[col], errors='coerce')
        parse_ratio = parsed.notna().mean()

        if parse_ratio == 0:
            continue

        name_score = 0

        if 'complete timestamp' in col_lower:
            name_score += 4

        if 'timestamp' in col_lower:
            name_score += 3

        if 'date' in col_lower:
            name_score += 1

        if pd.api.types.is_numeric_dtype(df[col]):
            name_score -= 5

        timestamp_candidates.append((name_score + parse_ratio, col))

    if timestamp_candidates:
        timestamp_col = max(timestamp_candidates)[1]

    if case_col is None:
        case_col = df.columns[0]

    if activity_col is None:
        activity_col = df.columns[1]

    return {
        'case': case_col,
        'activity': activity_col,
        'timestamp': timestamp_col,
        'resource': resource_col,
        'label': label_col,
    }


def preprocess_event_log(df, columns):
    case_col = columns['case']
    activity_col = columns['activity']
    timestamp_col = columns['timestamp']
    resource_col = columns['resource']

    cleaned = df.copy()
    cleaned = cleaned.dropna(subset=[case_col, activity_col])
    cleaned = cleaned.drop_duplicates()

    if timestamp_col is not None:
        cleaned['_parsed_timestamp'] = pd.to_datetime(
            cleaned[timestamp_col],
            errors='coerce'
        )
        cleaned = cleaned.dropna(subset=['_parsed_timestamp'])
        cleaned = cleaned.sort_values([case_col, '_parsed_timestamp'])
    else:
        cleaned['_parsed_timestamp'] = pd.NaT
        cleaned = cleaned.sort_values([case_col])

    cleaned['_case_id_norm'] = cleaned[case_col].astype(str)
    cleaned['_activity_norm'] = cleaned[activity_col].apply(_normalise_name)

    if resource_col is not None:
        cleaned['_resource_norm'] = cleaned[resource_col].apply(safe_resource_value)
    else:
        cleaned['_resource_norm'] = 'unknown'

    cleaned['_amount_numeric'] = 0.0
    cleaned['_expense_numeric'] = 0.0

    for col in cleaned.columns:
        col_lower = col.lower()

        if 'amount' in col_lower:
            cleaned['_amount_numeric'] = pd.to_numeric(
                cleaned[col],
                errors='coerce'
            ).fillna(0.0)

        elif 'expense' in col_lower:
            cleaned['_expense_numeric'] = pd.to_numeric(
                cleaned[col],
                errors='coerce'
            ).fillna(0.0)

    return cleaned.reset_index(drop=True)

