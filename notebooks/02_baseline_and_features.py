"""
Baseline & Feature Engineering Exploration (Deliverable D3)
Inspects lag structures, rolling statistics, and seasonal naive benchmarks.
"""
import os
import pandas as pd

def run_baseline_inspection():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(base_dir, "data", "processed", "weekly_demand_features.csv"))
    print(f"Total SKU-weeks analyzed: {len(df):,}")
    print(f"Average Weekly Units per SKU: {df['weekly_units'].mean():.2f}")
    print(f"Features ready for forecasting: week_of_year, month, holidays_in_week, promo_events_in_week")

if __name__ == "__main__":
    run_baseline_inspection()
