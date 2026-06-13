"""
STEP 2: Compute Latest Site Arrival & Order Dates
Uses predicted install hours + EOS + lead time.
No calendar yet — pure arithmetic planning anchor.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from step1_predict_install_hours import load_data, predict_all

EOS_BUFFER_DAYS = 30
WORKING_HOURS_PER_DAY = 8
TRANSPORT_SLA_DAYS = 5


def hours_to_days(hours, engineers=1):
    """Convert install hours to calendar days given engineer parallelism."""
    hours_per_day = WORKING_HOURS_PER_DAY * max(1, engineers)
    return max(1, int(np.ceil(hours / hours_per_day)))


def compute_planning_dates(df):
    df = df.copy()
    df['eos_date'] = pd.to_datetime(df['eos_date'])

    df['latest_active_date'] = df['eos_date'] - timedelta(days=EOS_BUFFER_DAYS)

    df['install_days'] = df.apply(
        lambda r: hours_to_days(r['predicted_install_hours'], r['engineers_available']), axis=1
    )

    df['latest_arrival_at_site'] = df['latest_active_date'] - df['install_days'].apply(lambda d: timedelta(days=d))

    df['latest_order_date'] = df.apply(
        lambda r: r['latest_arrival_at_site'] - timedelta(days=(TRANSPORT_SLA_DAYS if r['in_storage'] else r['vendor_lead_time_days'])),
        axis=1
    )

    today = datetime.today()
    df['days_until_order_deadline'] = (df['latest_order_date'] - today).dt.days

    df['urgency_flag'] = df['days_until_order_deadline'].apply(
        lambda d: 'OVERDUE' if d < 0 else ('URGENT' if d <= 30 else ('SOON' if d <= 90 else 'OK'))
    )

    return df


if __name__ == '__main__':
    df = predict_all()
    df = compute_planning_dates(df)

    cols = ['serial_number', 'device_type', 'eos_date', 'predicted_install_hours',
            'install_days', 'latest_arrival_at_site', 'latest_order_date',
            'days_until_order_deadline', 'urgency_flag']
    print(df[cols].sort_values('days_until_order_deadline').head(15).to_string(index=False))

    out = os.path.join(os.path.dirname(__file__), '..', 'data', 'planning_dates.csv')
    df.to_csv(out, index=False)
    print(f"\nPlanning dates saved -> {out}")
