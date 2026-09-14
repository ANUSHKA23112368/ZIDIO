"""
Exploratory Data Analysis & Quality Audit for NorthBay Living (Deliverable D2)
Analyzes sales distributions, Pareto velocity, category trends, and calendar seasonality.
"""
import os
import pandas as pd
import numpy as np

def run_eda():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sales = pd.read_csv(os.path.join(base_dir, "data", "raw", "sales_daily.csv"))
    skus = pd.read_csv(os.path.join(base_dir, "data", "processed", "clean_sku_master.csv"))
    weekly = pd.read_csv(os.path.join(base_dir, "data", "processed", "weekly_demand_features.csv"))
    
    print("=== NORTHBAY LIVING EDA SUMMARY ===")
    print(f"Total Transactions: {len(sales):,}")
    print(f"Active SKUs: {len(skus):,}")
    print(f"Catalog Categories: {skus['category'].unique().tolist()}")
    
    # Pareto calculation
    sku_revenue = sales.groupby("sku_id")["revenue"].sum().sort_values(ascending=False)
    top_15_pct = int(len(sku_revenue) * 0.15)
    top_rev_share = (sku_revenue.head(top_15_pct).sum() / sku_revenue.sum()) * 100
    print(f"Pareto Share: Top 15% SKUs generate {top_rev_share:.1f}% of total revenue")

if __name__ == "__main__":
    run_eda()
