"""
FULL PIPELINE — Run all 4 steps end to end.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 60)
print("DEVICE LIFECYCLE OPTIMIZER — FULL PIPELINE")
print("=" * 60)

print("\n[STEP 1] Training Ridge Regression for install hour prediction...")
from step1_predict_install_hours import train_and_evaluate, predict_all
pipeline, mae, r2 = train_and_evaluate()
df = predict_all()
print(f"         -> {len(df)} devices processed")

print("\n[STEP 2] Computing planning dates...")
from step2_compute_dates import compute_planning_dates
df = compute_planning_dates(df)
urgency_counts = df['urgency_flag'].value_counts()
print(f"         -> Urgency breakdown:\n{urgency_counts.to_string()}")

print("\n[STEP 3] Running MILP procurement optimization...")
from step3_milp_procurement import run_milp
result = run_milp(df)
if result:
    result_df, summary = result
    out3 = os.path.join(os.path.dirname(__file__), '..', 'data', 'procurement_plan.csv')
    result_df.to_csv(out3, index=False)

print("\n[STEP 4] Scheduling installations...")
from step4_schedule_installation import schedule
proc_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'procurement_plan.csv')
schedule_df = schedule(df, proc_path)
schedule_df.to_csv(
    os.path.join(os.path.dirname(__file__), '..', 'data', 'installation_schedule.csv'), index=False
)
print(f"         -> Schedule summary:\n{schedule_df['schedule_status'].value_counts().to_string()}")

print("\n" + "=" * 60)
print("PIPELINE COMPLETE. Results saved in data/")
print("=" * 60)
