import pandas as pd
from pathlib import Path
from utils.preprocessing import detect_event_log_columns

def run_eda(csv_path, output_dir):
    """
    Perform Exploratory Data Analysis (EDA) on an event log CSV file
    and save the report as a text file in output_dir.
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    try:
        df = pd.read_csv(csv_path, sep=None, engine='python')
    except Exception as e:
        # Fallback to default encoding or read if sep=None fails
        df = pd.read_csv(csv_path)

    # 2. Detect Columns
    columns = detect_event_log_columns(df)
    case_col = columns['case']
    activity_col = columns['activity']
    timestamp_col = columns['timestamp']
    resource_col = columns['resource']
    label_col = columns['label']

    report = []
    report.append("==================================================")
    report.append(f"EDA REPORT FOR: {csv_path.name}")
    report.append("==================================================")
    report.append("")

    # Basic Info
    report.append("----- DATA OVERVIEW -----")
    report.append(f"Total Rows (Events) : {len(df)}")
    report.append(f"Total Columns       : {len(df.columns)}")
    report.append(f"Detected Columns:")
    report.append(f"  - Case ID Column   : {case_col}")
    report.append(f"  - Activity Column  : {activity_col}")
    report.append(f"  - Timestamp Column : {timestamp_col if timestamp_col else 'Not Found'}")
    report.append(f"  - Resource Column  : {resource_col if resource_col else 'Not Found'}")
    report.append(f"  - Label Column     : {label_col if label_col else 'Not Found'}")
    report.append("")

    # Case Statistics
    if case_col in df.columns:
        unique_cases = df[case_col].nunique()
        report.append("----- CASE STATISTICS -----")
        report.append(f"Total Unique Cases  : {unique_cases}")
        
        events_per_case = df.groupby(case_col).size()
        report.append(f"Events Per Case:")
        report.append(f"  - Min             : {events_per_case.min()}")
        report.append(f"  - Max             : {events_per_case.max()}")
        report.append(f"  - Mean            : {events_per_case.mean():.2f}")
        report.append(f"  - Median          : {events_per_case.median():.2f}")
        report.append(f"  - Std Dev         : {events_per_case.std():.2f}")
        report.append("")

    # Case Duration Statistics (if Timestamp column is present)
    if timestamp_col and timestamp_col in df.columns:
        try:
            df_ts = df.copy()
            df_ts[timestamp_col] = pd.to_datetime(df_ts[timestamp_col], errors='coerce')
            df_ts = df_ts.dropna(subset=[timestamp_col])
            
            grouped_time = df_ts.groupby(case_col)[timestamp_col]
            # duration in days
            durations = (grouped_time.max() - grouped_time.min()).dt.total_seconds() / (24 * 3600)
            
            report.append("----- CASE DURATION STATISTICS (DAYS) -----")
            report.append(f"  - Min Duration    : {durations.min():.4f} days")
            report.append(f"  - Max Duration    : {durations.max():.4f} days")
            report.append(f"  - Mean Duration   : {durations.mean():.4f} days")
            report.append(f"  - Median Duration : {durations.median():.4f} days")
            report.append(f"  - Std Dev Duration: {durations.std():.4f} days")
            report.append("")
        except Exception as e:
            report.append("----- CASE DURATION STATISTICS (DAYS) -----")
            report.append(f"  Error calculating duration: {str(e)}")
            report.append("")

    # Activity Statistics
    if activity_col in df.columns:
        report.append("----- ACTIVITY STATISTICS -----")
        unique_activities = df[activity_col].nunique()
        report.append(f"Total Unique Activities: {unique_activities}")
        report.append("Top 10 Activities:")
        activity_counts = df[activity_col].value_counts()
        for idx, (act, count) in enumerate(activity_counts.head(10).items(), 1):
            pct = (count / len(df)) * 100
            report.append(f"  {idx}. {act:<30} : {count} ({pct:.2f}%)")
        report.append("")

    # Resource Statistics (if Resource column is present)
    if resource_col and resource_col in df.columns:
        report.append("----- RESOURCE STATISTICS -----")
        unique_resources = df[resource_col].nunique()
        report.append(f"Total Unique Resources: {unique_resources}")
        report.append("Top 10 Resources:")
        resource_counts = df[resource_col].value_counts()
        for idx, (res, count) in enumerate(resource_counts.head(10).items(), 1):
            pct = (count / len(df)) * 100
            report.append(f"  {idx}. {res:<30} : {count} ({pct:.2f}%)")
        report.append("")

    # Label Statistics (if Label column is present)
    if label_col and label_col in df.columns:
        report.append("----- LABEL DISTRIBUTION -----")
        label_counts = df[label_col].value_counts()
        for label, count in label_counts.items():
            pct = (count / len(df)) * 100
            report.append(f"  - {label:<15} : {count} ({pct:.2f}%)")
        report.append("")

    # Save to file
    report_text = "\n".join(report)
    output_path = output_dir / f"{csv_path.stem}_eda.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
        
    print(f"EDA report successfully saved to: {output_path}")
    return output_path
