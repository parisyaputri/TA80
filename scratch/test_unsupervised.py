# scratch/test_unsupervised.py

import pandas as pd
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
from utils.iterative_baseline import select_stable_baseline_cases

def test_unsupervised():
    csv_path = Path("data/p.csv")
    df = pd.read_csv(csv_path, sep=None, engine='python')
    columns = detect_event_log_columns(df)
    label_col = columns['label']
    
    cleaned_df = preprocess_event_log(df, columns)
    
    # ----------------------------------------------------
    # ROUND 1: Unsupervised Profile Learning
    # ----------------------------------------------------
    profile_grouped = cleaned_df.groupby('_case_id_norm', sort=False)
    cf_profile = learn_control_flow_profile(profile_grouped)
    resource_profile = learn_resource_profile(cleaned_df)
    case_states = replay_event_states(cleaned_df)
    
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
    
    # Drop labels to simulate unsupervised training
    unlabeled_feature_df = feature_df.drop(columns=['label'], errors='ignore')
    
    dt = DigitalTwin()
    dt.fit(unlabeled_feature_df, numeric_cols)
    dt.seed_states(case_states)
    
    ddc = DynamicDeclarativeConstraints()
    ddc.fit(dt, numeric_cols, cf_profile=cf_profile, resource_profile=resource_profile)
    
    mv_arm = MVARMiner()
    mv_arm.fit(unlabeled_feature_df)
    
    ib = IntelligentBody(dt, ddc, mv_arm)
    # Calibrate without labels (passing unlabeled df)
    ib.calibrate_weights(unlabeled_feature_df, numeric_cols)
    
    scoring_df = feature_df.copy()
    precomputed_arm = mv_arm.score_dataframe(scoring_df)
    scoring_df['_pre_arm_score'] = precomputed_arm['arm_score']
    scoring_df['_pre_arm_rules_hit'] = precomputed_arm['arm_rules_hit']
    scoring_df['_pre_violated_arm_rules'] = precomputed_arm['violated_arm_rules']
    
    initial_scored = ib.score_all(scoring_df)
    
    # Analyze Round 1 scores
    temp_df = feature_df.copy()
    temp_df['initial_anomaly_score'] = initial_scored['anomaly_score']
    temp_df['initial_ddc_score'] = initial_scored['ddc_score']
    temp_df['initial_z_score'] = initial_scored['z_score']
    temp_df['initial_arm_score'] = initial_scored['arm_score']
    temp_df['initial_br_score'] = initial_scored['br_score']
    
    print("\n===== ROUND 1 MEAN SCORES BY LABEL =====")
    print(temp_df.groupby('label')[['initial_anomaly_score', 'initial_ddc_score', 'initial_z_score', 'initial_arm_score', 'initial_br_score']].mean().T)
    
    # ----------------------------------------------------
    # ROUND 2: Refit on Unsupervised Stable Baseline Cases
    # ----------------------------------------------------
    stable_case_ids = select_stable_baseline_cases(initial_scored)
    
    stable_events = cleaned_df[
        cleaned_df['_case_id_norm'].astype(str).isin(stable_case_ids)
    ].copy()
    
    cf_profile_round2 = learn_control_flow_profile(stable_events.groupby('_case_id_norm', sort=False))
    resource_profile_round2 = learn_resource_profile(stable_events)
    
    baseline_feature_df = feature_df[
        feature_df['case_id'].astype(str).isin(stable_case_ids)
    ].drop(columns=['label'], errors='ignore')
    
    dt_round2 = DigitalTwin()
    dt_round2.fit(baseline_feature_df, numeric_cols)
    dt_round2.seed_states(case_states)
    
    ddc_round2 = DynamicDeclarativeConstraints()
    ddc_round2.fit(dt_round2, numeric_cols, cf_profile=cf_profile_round2, resource_profile=resource_profile_round2)
    
    mv_arm_round2 = MVARMiner()
    mv_arm_round2.fit(baseline_feature_df)
    
    ib_round2 = IntelligentBody(dt_round2, ddc_round2, mv_arm_round2)
    ib_round2.calibrate_weights(baseline_feature_df, numeric_cols)
    
    scoring_df_round2 = feature_df.copy()
    precomputed_arm_round2 = mv_arm_round2.score_dataframe(scoring_df_round2)
    scoring_df_round2['_pre_arm_score'] = precomputed_arm_round2['arm_score']
    scoring_df_round2['_pre_arm_rules_hit'] = precomputed_arm_round2['arm_rules_hit']
    scoring_df_round2['_pre_violated_arm_rules'] = precomputed_arm_round2['violated_arm_rules']
    
    final_scored = ib_round2.score_all(scoring_df_round2)
    
    y_true = feature_df['label'].eq('deviant').astype(int)
    
    print("\n===== ROUND 2 UNSUPERVISED COMPONENT AUC =====")
    for col in ['ddc_score', 'z_score', 'arm_score', 'br_score', 'anomaly_score']:
        auc = roc_auc_score(y_true, final_scored[col])
        print(f"{col:<15} : {auc:.4f}")

    # ----------------------------------------------------
    # SIMULATION: Realistic Contamination Rate (~6.5% anomalies)
    # ----------------------------------------------------
    regular_cases = feature_df[feature_df['label'] == 'regular']
    deviant_cases = feature_df[feature_df['label'] == 'deviant'].sample(5, random_state=42)
    sim_feature_df = pd.concat([regular_cases, deviant_cases], ignore_index=True)
    
    sim_case_ids = set(sim_feature_df['case_id'].astype(str))
    sim_cleaned_df = cleaned_df[cleaned_df['_case_id_norm'].astype(str).isin(sim_case_ids)].copy()
    
    # Run Round 1 on simulated dataset
    sim_profile_grouped = sim_cleaned_df.groupby('_case_id_norm', sort=False)
    sim_cf_profile = learn_control_flow_profile(sim_profile_grouped)
    sim_resource_profile = learn_resource_profile(sim_cleaned_df)
    sim_case_states = replay_event_states(sim_cleaned_df)
    
    sim_feature_df = build_case_features(
        sim_cleaned_df, sim_cf_profile, sim_resource_profile, sim_case_states, label_col
    )
    
    sim_unlabeled_df = sim_feature_df.drop(columns=['label'], errors='ignore')
    
    sim_dt = DigitalTwin()
    sim_dt.fit(sim_unlabeled_df, numeric_cols)
    sim_dt.seed_states(sim_case_states)
    
    sim_ddc = DynamicDeclarativeConstraints()
    sim_ddc.fit(sim_dt, numeric_cols, cf_profile=sim_cf_profile, resource_profile=sim_resource_profile)
    
    sim_mv_arm = MVARMiner()
    sim_mv_arm.fit(sim_unlabeled_df)
    
    sim_ib = IntelligentBody(sim_dt, sim_ddc, sim_mv_arm)
    sim_ib.calibrate_weights(sim_unlabeled_df, numeric_cols)
    
    sim_scoring_df = sim_feature_df.copy()
    sim_precomputed_arm = sim_mv_arm.score_dataframe(sim_scoring_df)
    sim_scoring_df['_pre_arm_score'] = sim_precomputed_arm['arm_score']
    sim_scoring_df['_pre_arm_rules_hit'] = sim_precomputed_arm['arm_rules_hit']
    sim_scoring_df['_pre_violated_arm_rules'] = sim_precomputed_arm['violated_arm_rules']
    
    sim_initial_scored = sim_ib.score_all(sim_scoring_df)
    
    # Round 2 refit
    sim_stable_case_ids = select_stable_baseline_cases(sim_initial_scored)
    sim_stable_events = sim_cleaned_df[sim_cleaned_df['_case_id_norm'].astype(str).isin(sim_stable_case_ids)].copy()
    
    sim_cf_profile_r2 = learn_control_flow_profile(sim_stable_events.groupby('_case_id_norm', sort=False))
    sim_res_profile_r2 = learn_resource_profile(sim_stable_events)
    
    sim_baseline_feature_df = sim_feature_df[
        sim_feature_df['case_id'].astype(str).isin(sim_stable_case_ids)
    ].drop(columns=['label'], errors='ignore')
    
    sim_dt_r2 = DigitalTwin()
    sim_dt_r2.fit(sim_baseline_feature_df, numeric_cols)
    sim_dt_r2.seed_states(sim_case_states)
    
    sim_ddc_r2 = DynamicDeclarativeConstraints()
    sim_ddc_r2.fit(sim_dt_r2, numeric_cols, cf_profile=sim_cf_profile_r2, resource_profile=sim_res_profile_r2)
    
    sim_mv_arm_r2 = MVARMiner()
    sim_mv_arm_r2.fit(sim_baseline_feature_df)
    
    sim_ib_r2 = IntelligentBody(sim_dt_r2, sim_ddc_r2, sim_mv_arm_r2)
    sim_ib_r2.calibrate_weights(sim_baseline_feature_df, numeric_cols)
    
    sim_scoring_df_r2 = sim_feature_df.copy()
    sim_precomputed_arm_r2 = sim_mv_arm_r2.score_dataframe(sim_scoring_df_r2)
    sim_scoring_df_r2['_pre_arm_score'] = sim_precomputed_arm_r2['arm_score']
    sim_scoring_df_r2['_pre_arm_rules_hit'] = sim_precomputed_arm_r2['arm_rules_hit']
    sim_scoring_df_r2['_pre_violated_arm_rules'] = sim_precomputed_arm_r2['violated_arm_rules']
    
    sim_final_scored = sim_ib_r2.score_all(sim_scoring_df_r2)
    
    sim_y_true = sim_feature_df['label'].eq('deviant').astype(int)
    
    print("\n===== SIMULATED REALISTIC CONTAMINATION COMPONENT AUC =====")
    for col in ['ddc_score', 'z_score', 'arm_score', 'br_score', 'anomaly_score']:
        auc = roc_auc_score(sim_y_true, sim_final_scored[col])
        print(f"{col:<15} : {auc:.4f}")

if __name__ == '__main__':
    test_unsupervised()
