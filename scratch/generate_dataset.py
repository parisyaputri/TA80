# scratch/generate_dataset.py

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_loan_dataset(output_path="data/synthetic_loan_process.csv", num_cases=2000, anomaly_rate=0.10):
    np.random.seed(42)
    random.seed(42)
    
    activities_list = [
        "Submit Application",
        "Verify Documents",
        "Assess Risk",
        "Approve Loan",
        "Disburse Funds",
        "Close Case"
    ]
    
    clerks = ["clerk_john", "clerk_sarah", "clerk_dave"]
    underwriters = ["underwriter_mary", "underwriter_paul", "underwriter_lucas"]
    managers = ["manager_alice", "manager_tom"]
    finances = ["finance_bob", "finance_jane"]
    
    events = []
    base_time = datetime(2026, 1, 1, 8, 0, 0)
    
    num_anomalous = int(num_cases * anomaly_rate)
    num_normal = num_cases - num_anomalous
    
    # Generate case labels and anomaly assignments
    case_types = ["regular"] * num_normal
    anomaly_types = ["None"] * num_normal
    
    # 25% of anomalies: Control-Flow
    # 25% of anomalies: Temporal
    # 25% of anomalies: Resource
    # 25% of anomalies: Mixed
    a_types = ["control-flow", "temporal", "resource", "mixed"]
    for i in range(num_anomalous):
        case_types.append("deviant")
        anomaly_types.append(a_types[i % len(a_types)])
        
    # Shuffle case order
    combined = list(zip(case_types, anomaly_types))
    random.shuffle(combined)
    case_types, anomaly_types = zip(*combined)
    
    for case_idx in range(num_cases):
        case_id = f"CASE_{1000 + case_idx}"
        label = case_types[case_idx]
        anomaly = anomaly_types[case_idx]
        
        # Start time of the case
        current_time = base_time + timedelta(
            days=random.randint(0, 100),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        
        amount = float(np.round(np.random.exponential(5000) + 1000, 2))
        expense = float(np.round(amount * 0.01 + np.random.normal(50, 10), 2))
        expense = max(10.0, expense)
        
        if label == "regular":
            # Normal well-behaved process flow
            path = activities_list.copy()
            
            for idx, act in enumerate(path):
                # Timing
                if idx > 0:
                    delay_hours = random.uniform(1.0, 12.0)
                    current_time += timedelta(hours=delay_hours)
                
                # Resources
                if act == "Submit Application":
                    res = f"customer_{random.randint(100, 999)}"
                elif act == "Verify Documents":
                    res = random.choice(clerks)
                elif act == "Assess Risk":
                    res = random.choice(underwriters)
                elif act == "Approve Loan":
                    res = random.choice(managers)
                elif act == "Disburse Funds":
                    res = random.choice(finances)
                else:
                    res = "system_auto"
                    
                events.append({
                    "Case ID": case_id,
                    "Activity": act,
                    "Resource": res,
                    "Timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "amount": amount,
                    "expense": expense,
                    "label": label,
                    "anomaly_type": anomaly
                })
                
        else: # deviant cases
            if anomaly == "control-flow":
                # Anomaly: skipped Manager Approval or wrong order
                sub_type = random.choice(["skipped_approval", "wrong_order"])
                
                if sub_type == "skipped_approval":
                    # Skip 'Approve Loan'
                    path = ["Submit Application", "Verify Documents", "Assess Risk", "Disburse Funds", "Close Case"]
                    for idx, act in enumerate(path):
                        if idx > 0:
                            current_time += timedelta(hours=random.uniform(1.0, 12.0))
                        
                        if act == "Submit Application":
                            res = f"customer_{random.randint(100, 999)}"
                        elif act == "Verify Documents":
                            res = random.choice(clerks)
                        elif act == "Assess Risk":
                            res = random.choice(underwriters)
                        elif act == "Disburse Funds":
                            res = random.choice(finances)
                        else:
                            res = "system_auto"
                            
                        events.append({
                            "Case ID": case_id,
                            "Activity": act,
                            "Resource": res,
                            "Timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                            "amount": amount,
                            "expense": expense,
                            "label": label,
                            "anomaly_type": anomaly
                        })
                else: # wrong_order
                    # Disburse before Approve
                    path = ["Submit Application", "Verify Documents", "Assess Risk", "Disburse Funds", "Approve Loan", "Close Case"]
                    for idx, act in enumerate(path):
                        if idx > 0:
                            current_time += timedelta(hours=random.uniform(1.0, 12.0))
                        
                        if act == "Submit Application":
                            res = f"customer_{random.randint(100, 999)}"
                        elif act == "Verify Documents":
                            res = random.choice(clerks)
                        elif act == "Assess Risk":
                            res = random.choice(underwriters)
                        elif act == "Disburse Funds":
                            res = random.choice(finances)
                        elif act == "Approve Loan":
                            res = random.choice(managers)
                        else:
                            res = "system_auto"
                            
                        events.append({
                            "Case ID": case_id,
                            "Activity": act,
                            "Resource": res,
                            "Timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                            "amount": amount,
                            "expense": expense,
                            "label": label,
                            "anomaly_type": anomaly
                        })
                        
            elif anomaly == "temporal":
                # Anomaly: severe delay in document verification or risk assessment
                path = activities_list.copy()
                delay_idx = random.choice([1, 2, 3]) # which step gets delayed
                
                for idx, act in enumerate(path):
                    if idx > 0:
                        if idx == delay_idx:
                            # Severe delay: 15 to 30 days
                            delay_days = random.uniform(15.0, 30.0)
                            current_time += timedelta(days=delay_days)
                        else:
                            current_time += timedelta(hours=random.uniform(1.0, 12.0))
                    
                    if act == "Submit Application":
                        res = f"customer_{random.randint(100, 999)}"
                    elif act == "Verify Documents":
                        res = random.choice(clerks)
                    elif act == "Assess Risk":
                        res = random.choice(underwriters)
                    elif act == "Approve Loan":
                        res = random.choice(managers)
                    elif act == "Disburse Funds":
                        res = random.choice(finances)
                    else:
                        res = "system_auto"
                        
                    events.append({
                        "Case ID": case_id,
                        "Activity": act,
                        "Resource": res,
                        "Timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "amount": amount,
                        "expense": expense,
                        "label": label,
                        "anomaly_type": anomaly
                    })
                    
            elif anomaly == "resource":
                # Anomaly: low level Clerk approving loan, or segregation of duties violation
                path = activities_list.copy()
                sub_type = random.choice(["clerk_approves", "same_person"])
                
                chosen_clerk = random.choice(clerks)
                
                for idx, act in enumerate(path):
                    if idx > 0:
                        current_time += timedelta(hours=random.uniform(1.0, 12.0))
                    
                    if act == "Submit Application":
                        res = f"customer_{random.randint(100, 999)}"
                    elif act == "Verify Documents":
                        res = chosen_clerk
                    elif act == "Assess Risk":
                        res = random.choice(underwriters)
                    elif act == "Approve Loan":
                        if sub_type == "clerk_approves":
                            # Clerk does the approval (unauthorized role)
                            res = chosen_clerk
                        else:
                            # The same underwriter who did risk also does approval (segregation of duties)
                            # Actually, we can reuse the underwriter resource for manager role
                            res = "underwriter_mary"
                    elif act == "Disburse Funds":
                        res = random.choice(finances)
                    else:
                        res = "system_auto"
                        
                    events.append({
                        "Case ID": case_id,
                        "Activity": act,
                        "Resource": res,
                        "Timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "amount": amount,
                        "expense": expense,
                        "label": label,
                        "anomaly_type": anomaly
                    })
                    
            else: # mixed
                # Anomaly: skipped risk assessment, severe delay in disbursement, approved by clerk
                path = ["Submit Application", "Verify Documents", "Approve Loan", "Disburse Funds", "Close Case"]
                chosen_clerk = random.choice(clerks)
                
                for idx, act in enumerate(path):
                    if idx > 0:
                        if act == "Disburse Funds":
                            # Severe delay
                            current_time += timedelta(days=random.uniform(20.0, 35.0))
                        else:
                            current_time += timedelta(hours=random.uniform(1.0, 12.0))
                    
                    if act == "Submit Application":
                        res = f"customer_{random.randint(100, 999)}"
                    elif act == "Verify Documents":
                        res = chosen_clerk
                    elif act == "Approve Loan":
                        res = chosen_clerk # Clerk approves
                    elif act == "Disburse Funds":
                        res = random.choice(finances)
                    else:
                        res = "system_auto"
                        
                    events.append({
                        "Case ID": case_id,
                        "Activity": act,
                        "Resource": res,
                        "Timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "amount": amount,
                        "expense": expense,
                        "label": label,
                        "anomaly_type": anomaly
                    })

    # Sort all events chronologically to simulate a real event-stream log
    df = pd.DataFrame(events)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values(by=["Timestamp"]).reset_index(drop=True)
    df["Timestamp"] = df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    df.to_csv(output_path, index=False, sep=";")
    print(f"Generated synthetic dataset with {len(df)} events, {num_cases} cases ({num_normal} normal, {num_anomalous} anomalous) at {output_path}")

if __name__ == '__main__':
    generate_loan_dataset()
