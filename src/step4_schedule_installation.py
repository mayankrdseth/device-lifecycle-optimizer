"""
STEP 4: Site-Level Installation Scheduling
Assigns serial numbers to actual site windows using calendar + engineer availability.
Uses a greedy priority-based scheduler (MILP extension possible).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from step1_predict_install_hours import load_data, predict_all
from step2_compute_dates import compute_planning_dates, hours_to_days

PRIORITY_ORDER = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
WORKING_HOURS_PER_DAY = 8


class SiteCalendar:
    """Tracks engineer-hours available per day per site."""

    def __init__(self, site_id, engineers, start_date, end_date,
                 weekend_allowed=False, daily_hours=8):
        self.site_id = site_id
        self.engineers = engineers
        self.daily_capacity = engineers * daily_hours
        self.weekend_allowed = weekend_allowed
        self.capacity = {}
        d = start_date
        while d <= end_date:
            if weekend_allowed or d.weekday() < 5:
                self.capacity[d] = self.daily_capacity
            d += timedelta(days=1)

    def find_slot(self, required_hours, earliest_start):
        """Find first slot where required_hours can be completed."""
        accumulated = 0
        slot_start = None
        for day in sorted(self.capacity.keys()):
            if day < earliest_start:
                continue
            if slot_start is None:
                slot_start = day
            available = self.capacity[day]
            use = min(available, required_hours - accumulated)
            accumulated += use
            if accumulated >= required_hours:
                return slot_start, day
        return None, None

    def book(self, required_hours, start_date, finish_date):
        """Deduct hours from calendar."""
        remaining = required_hours
        for day in sorted(self.capacity.keys()):
            if day < start_date or day > finish_date:
                continue
            use = min(self.capacity[day], remaining)
            self.capacity[day] -= use
            remaining -= use
            if remaining <= 0:
                break


def build_site_calendars(df, planning_start, planning_end):
    calendars = {}
    for site_id, grp in df.groupby('site_id'):
        engineers = int(grp['engineers_available'].median())
        site_type = grp['site_type'].iloc[0]
        weekend_allowed = site_type == 'Data Center'
        calendars[site_id] = SiteCalendar(
            site_id, engineers, planning_start, planning_end,
            weekend_allowed=weekend_allowed
        )
    return calendars


def schedule(df, procurement_plan_path=None):
    df = df.copy()

    if procurement_plan_path and os.path.exists(procurement_plan_path):
        proc = pd.read_csv(procurement_plan_path)[['serial_number', 'procurement_month']]
        df = df.merge(proc, on='serial_number', how='left')
    else:
        df['procurement_month'] = None

    df['priority_order'] = df['priority'].map(PRIORITY_ORDER).fillna(3)
    df = df.sort_values(['priority_order', 'days_until_order_deadline'])

    planning_start = datetime.today().replace(day=1)
    planning_end = planning_start + timedelta(days=365)

    calendars = build_site_calendars(df, planning_start, planning_end)

    results = []
    for _, row in df.iterrows():
        site_id = row['site_id']
        cal = calendars.get(site_id)
        if cal is None:
            continue

        if pd.notna(row.get('latest_arrival_at_site')):
            arr = pd.Timestamp(row['latest_arrival_at_site'])
            earliest = max(planning_start, arr - timedelta(days=30))
        else:
            earliest = planning_start

        required_hours = row['predicted_install_hours']
        slot_start, slot_end = cal.find_slot(required_hours, earliest)

        if slot_start:
            cal.book(required_hours, slot_start, slot_end)
            results.append({
                **row.to_dict(),
                'scheduled_start': slot_start.strftime('%Y-%m-%d'),
                'scheduled_end': slot_end.strftime('%Y-%m-%d'),
                'schedule_status': 'SCHEDULED'
            })
        else:
            results.append({
                **row.to_dict(),
                'scheduled_start': None,
                'scheduled_end': None,
                'schedule_status': 'UNSCHEDULABLE'
            })

    return pd.DataFrame(results)


if __name__ == '__main__':
    df = predict_all()
    df = compute_planning_dates(df)
    proc_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'procurement_plan.csv')
    schedule_df = schedule(df, proc_path)

    cols = ['serial_number', 'device_type', 'site_id', 'priority',
            'predicted_install_hours', 'scheduled_start', 'scheduled_end', 'schedule_status']
    print(schedule_df[cols].head(20).to_string(index=False))

    summary = schedule_df['schedule_status'].value_counts()
    print(f"\nSchedule summary:\n{summary.to_string()}")

    out = os.path.join(os.path.dirname(__file__), '..', 'data', 'installation_schedule.csv')
    schedule_df.to_csv(out, index=False)
    print(f"\nInstallation schedule saved -> {out}")
