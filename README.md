# Device Lifecycle Optimizer

A modular ML + MILP pipeline to plan device replacements at scale — predicting install effort, computing procurement deadlines, and scheduling installations across sites.

---

## Problem Statement

Telecom and enterprise networks run thousands of devices (routers, switches, firewalls, servers) that have an **End-of-Support (EOS)** date. Missing EOS means security risk and vendor non-compliance. The goal is to:

1. Predict how long each device takes to install.
2. Compute when each device must be procured.
3. Batch orders monthly to respect budget constraints.
4. Schedule installations across sites given engineer capacity and calendar windows.

---

## Pipeline Overview

```
[Serial Number Data]
        │
        ▼
┌──────────────────────────────────┐
│  STEP 1: Ridge Regression        │
│  Predict install_hours per SN    │
│  No calendar info used here      │
└──────────────────┬───────────────┘
                   │ predicted_install_hours
                   ▼
┌──────────────────────────────────┐
│  STEP 2: Date Arithmetic         │
│  latest_active_date = EOS - 30d  │
│  latest_arrival = active - hours │
│  latest_order = arrival - lead   │
└──────────────────┬───────────────┘
                   │ planning anchors
                   ▼
┌──────────────────────────────────┐
│  STEP 3: MILP Procurement        │
│  Decide: which SN in which month │
│  Respect: budget, EOS deadlines  │
│  Minimize: cost + EOS violations │
└──────────────────┬───────────────┘
                   │ procurement_month per SN
                   ▼
┌──────────────────────────────────┐
│  STEP 4: Installation Scheduler  │
│  Assign: SN → site window        │
│  Respect: engineer capacity      │
│           site calendar          │
│           parallel installs      │
│  Priority: Critical > High > ... │
└──────────────────────────────────┘
```

---

## Why This Separation?

| Concern | Where handled | Why |
|---|---|---|
| Install effort estimation | Step 1 (ML) | Regression works best on clean features; calendar adds noise |
| Procurement deadline | Step 2 (rules) | Pure arithmetic, deterministic, explainable |
| Monthly batching + budget | Step 3 (MILP) | Optimization problem with hard constraints |
| Engineer allocation + windows | Step 4 (scheduling) | Calendar is a feasibility check, not a regression target |

---

## Key Design Decisions

### Why Ridge Regression?
- Device features are correlated (complexity ↔ vendor ↔ site type).
- Ridge handles multicollinearity by shrinking coefficients toward zero.
- Relationships are mostly additive — linear is interpretable and sufficient.

### Why MILP for procurement?
- Monthly batching is a combinatorial decision.
- Hard constraint: no device can cross EOS.
- Soft constraint: minimize cost, balance monthly spend.
- MILP gives an optimal/near-optimal solution with solver guarantees.

### Why separate the calendar?
- Predicted install hours = **effort** (calendar-free).
- Actual install window = **scheduled effort** (calendar-dependent).
- Mixing them would make the regression target unstable across sites.

### When would Gradient Boosting be better?
- If install time depends on nonlinear thresholds (e.g., benefit of extra engineer drops after 4).
- If interactions between site type and device type are strong.
- If the dataset grows large enough to exploit tree-based patterns.

---

## Project Structure

```
device-lifecycle-optimizer/
├── data/
│   ├── device_lifecycle_data.csv    # Synthetic dataset (500 serial numbers)
│   ├── planning_dates.csv           # Output of Step 2
│   ├── procurement_plan.csv         # Output of Step 3
│   └── installation_schedule.csv   # Output of Step 4
├── src/
│   ├── step1_predict_install_hours.py
│   ├── step2_compute_dates.py
│   ├── step3_milp_procurement.py
│   ├── step4_schedule_installation.py
│   └── run_pipeline.py              # Run all steps end-to-end
├── notebooks/
│   └── (exploratory notebooks — add your own)
├── requirements.txt
└── README.md
```

---

## Getting Started

```bash
# Clone
git clone https://github.com/mayankrdseth/device-lifecycle-optimizer.git
cd device-lifecycle-optimizer

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python src/run_pipeline.py
```

### Run individual steps

```bash
python src/step1_predict_install_hours.py   # Train regression model
python src/step2_compute_dates.py           # Compute planning dates
python src/step3_milp_procurement.py        # MILP procurement plan
python src/step4_schedule_installation.py   # Site installation schedule
```

---

## Dataset Schema

| Column | Type | Description |
|---|---|---|
| `serial_number` | str | Unique device identifier |
| `device_type` | str | Router, Switch, Firewall, Server, etc. |
| `vendor` | str | Cisco, Juniper, Dell, etc. |
| `site_id` | str | Site where device will be installed |
| `site_type` | str | Data Center, Office, Branch, Remote |
| `engineers_available` | int | Engineers dedicated to this site |
| `available_window_days` | int | Days available for installation work |
| `device_complexity_score` | int | 1–5 complexity rating |
| `purchase_date` | date | Original purchase date |
| `eos_date` | date | End-of-Support deadline |
| `vendor_lead_time_days` | int | Days from order to site delivery |
| `in_storage` | bool | Whether device is already in inventory |
| `priority` | str | Critical, High, Medium, Low |
| `install_hours_actual` | float | Historical install hours (training label) |

---

## Output Files

| File | Description |
|---|---|
| `planning_dates.csv` | Per-SN: predicted hours, latest order/arrival dates, urgency flag |
| `procurement_plan.csv` | Per-SN: assigned procurement month or MISSED flag |
| `installation_schedule.csv` | Per-SN: scheduled start/end date at site |

---

## Extensions & Next Steps

- [ ] Add Gradient Boosting model in Step 1 and compare with Ridge.
- [ ] Replace greedy Step 4 scheduler with a full MILP (parallel-machine scheduling).
- [ ] Add real vendor API integration for live lead times.
- [ ] Build a Streamlit dashboard to visualize the Gantt chart per site.
- [ ] Add stochastic lead time handling (lead time distributions instead of point estimates).
- [ ] Add cost catalog for accurate device pricing.

---

## Dependencies

- `scikit-learn` — Ridge regression pipeline
- `pulp` — MILP solver (CBC backend)
- `pandas`, `numpy` — data processing
- `matplotlib`, `seaborn` — visualization

---

## License

MIT
