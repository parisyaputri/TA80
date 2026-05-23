# scratch/analyze_p.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

from models.digital_twin import DigitalTwin
from models.ddc import DynamicDeclarativeConstraints
from models.mv_arm import MVARMiner
from models.intelligent_body import IntelligentBody

from utils.preprocessing import detect_event_log_columns, preprocess_event_log
from utils.feature_engineering import (
    learn_control_flow_profile,
    learn_resource_profile,
    replay_event_states,
    build_case_features
)

def analyze():
    csv_path = Path("data/p.csv")
    df = pd.read_csv(csv_path, sep=None, engine='python')
    columns = detect_event_log_columns(df)
    label_col = columns['label']
    
    cleaned_df = preprocess_event_log(df, columns)
    
    # Learn profiles
    profile_grouped = cleaned_df.groupby('_case_id_norm', sort=False)
    cf_profile = learn_control_flow_profile(profile_grouped)
    resource_profile = learn_resource_profile(cleaned_df)
    case_states = replay_event_states(cleaned_df)
    
    # Features
    feature_df = build_case_features(
        cleaned_df, cf_profile, resource_profile, case_states, label_col
    )
    
    numeric_cols = [
        'cf_n_events', 'cf_seq_violations', 'cf_missing_steps', 'cf_duplicate_steps',
        'cf_wrong_order_ratio', 'temp_total_hrs', 'temp_max_step_hrs', 'temp_std_step_hrs',
        'res_n_resources', 'res_single_resource', 'res_many_resources',
        'res_dominant_resource_ratio', 'res_unusual_activity_count',
        'res_unusual_activity_ratio', 'res_workload_share', 'res_resource_rarity',
        'amount', 'expense'
    ]
    
    feature_for_learning = feature_df.drop(columns=['label'], errors='ignore')
    
    # Fit models
    dt = DigitalTwin()
    dt.fit(feature_for_learning, numeric_cols)
    dt.seed_states(case_states)
    
    ddc = DynamicDeclarativeConstraints()
    ddc.fit(dt, numeric_cols, cf_profile=cf_profile, resource_profile=resource_profile)
    
    mv_arm = MVARMiner()
    mv_arm.fit(feature_for_learning)
    
    ib = IntelligentBody(dt, ddc, mv_arm)
    ib.calibrate_weights(feature_for_learning, numeric_cols)
    
    # Score
    scoring_df = feature_df.copy()
    precomputed_arm = mv_arm.score_dataframe(scoring_df)
    scoring_df['_pre_arm_score'] = precomputed_arm['arm_score']
    scoring_df['_pre_arm_rules_hit'] = precomputed_arm['arm_rules_hit']
    scoring_df['_pre_violated_arm_rules'] = precomputed_arm['violated_arm_rules']
    
    result_df = ib.score_all(scoring_df)
    final_df = pd.merge(feature_df, result_df, on='case_id')
    
    y_true = final_df['label'].eq('deviant').astype(int)
    
    print("\n===== COMPONENT AUC-ROC =====")
    for col in ['ddc_score', 'z_score', 'arm_score', 'br_score', 'anomaly_score']:
        auc = roc_auc_score(y_true, final_df[col])
        print(f"{col:<15} : {auc:.4f}")
        
    print("\nCalibrated weights:")
    print(ib.component_weights)

if __name__ == '__main__':
    analyze()
