"""
STEP 3: MILP — Monthly Procurement Batching
Decides which serial numbers get ordered in which month.
Minimizes cost + EOS risk penalty while respecting monthly budget.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

try:
    import pulp
except ImportError:
    print("Install pulp: pip install pulp")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(__file__))
from step1_predict_install_hours import load_data, predict_all
from step2_compute_dates import compute_planning_dates

MONTHLY_BUDGET = 500_000
COST_PER_DEVICE = {
    'Router': 15000, 'Switch': 8000, 'Firewall': 20000,
    'Access Point': 2000, 'Server': 40000, 'Load Balancer': 25000
}
HORIZON_MONTHS = 12
OVERDUE_PENALTY = 1_000_000


def get_planning_months(n=HORIZON_MONTHS):
    today = datetime.today().replace(day=1)
    return [today + timedelta(days=30*i) for i in range(n)]


def run_milp(df, horizon_months=HORIZON_MONTHS):
    """
    Decision variable: x[i][m] = 1 if serial number i is procured in month m.
    Objective: minimize total cost + EOS violation penalty.
    """
    months = get_planning_months(horizon_months)
    month_labels = [m.strftime('%Y-%m') for m in months]

    proc_df = df[df['days_until_order_deadline'] <= horizon_months * 30].copy().reset_index(drop=True)
    n_devices = len(proc_df)

    if n_devices == 0:
        print("[Step 3] No devices need procurement in the planning horizon.")
        return None

    print(f"[Step 3] Optimizing procurement for {n_devices} devices over {horizon_months} months...")

    prob = pulp.LpProblem("DeviceProcurement", pulp.LpMinimize)

    x = [[pulp.LpVariable(f"x_{i}_{m}", cat='Binary') for m in range(horizon_months)]
         for i in range(n_devices)]

    slack = [pulp.LpVariable(f"slack_{i}", cat='Binary') for i in range(n_devices)]

    device_costs = [COST_PER_DEVICE.get(proc_df.loc[i, 'device_type'], 10000) for i in range(n_devices)]

    prob += (
        pulp.lpSum(device_costs[i] * x[i][m] for i in range(n_devices) for m in range(horizon_months))
        + pulp.lpSum(OVERDUE_PENALTY * slack[i] for i in range(n_devices))
    )

    # Each device ordered exactly once OR slack=1
    for i in range(n_devices):
        prob += pulp.lpSum(x[i][m] for m in range(horizon_months)) + slack[i] == 1

    # Monthly budget constraint
    for m in range(horizon_months):
        prob += pulp.lpSum(device_costs[i] * x[i][m] for i in range(n_devices)) <= MONTHLY_BUDGET

    # Cannot order after deadline
    for i in range(n_devices):
        deadline = proc_df.loc[i, 'latest_order_date']
        if pd.isna(deadline):
            continue
        deadline = pd.Timestamp(deadline)
        for m in range(horizon_months):
            if months[m] > deadline:
                prob += x[i][m] == 0

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    status = pulp.LpStatus[prob.status]
    print(f"[Step 3] Solver status: {status}")

    results = []
    for i in range(n_devices):
        assigned_month = None
        for m in range(horizon_months):
            if pulp.value(x[i][m]) and pulp.value(x[i][m]) > 0.5:
                assigned_month = month_labels[m]
                break
        missed = pulp.value(slack[i]) > 0.5 if pulp.value(slack[i]) is not None else True
        results.append({
            **proc_df.loc[i].to_dict(),
            'procurement_month': assigned_month if not missed else 'MISSED',
            'eos_violation': missed
        })

    result_df = pd.DataFrame(results)
    monthly_summary = result_df[result_df['procurement_month'] != 'MISSED'].groupby('procurement_month').agg(
        devices_ordered=('serial_number', 'count'),
        total_cost=('device_type', lambda x: sum(COST_PER_DEVICE.get(v, 10000) for v in x))
    ).reset_index()

    print("\n[Step 3] Monthly procurement plan:")
    print(monthly_summary.to_string(index=False))

    missed_count = result_df['eos_violation'].sum()
    print(f"\n[Step 3] EOS violations: {int(missed_count)} devices cannot be scheduled within budget/horizon.")

    return result_df, monthly_summary


if __name__ == '__main__':
    df = predict_all()
    df = compute_planning_dates(df)
    result = run_milp(df)
    if result:
        result_df, summary = result
        out = os.path.join(os.path.dirname(__file__), '..', 'data', 'procurement_plan.csv')
        result_df.to_csv(out, index=False)
        print(f"\nProcurement plan saved -> {out}")
