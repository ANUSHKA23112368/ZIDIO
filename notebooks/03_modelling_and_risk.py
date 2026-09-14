"""
Modelling & Risk Decisioning Analysis (Deliverables D3 & D4)
Inspects the 2x2 decisioning matrix, stockout risks, and locked capital.
"""
import os
import pandas as pd

def run_risk_inspection():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    grid = pd.read_csv(os.path.join(base_dir, "data", "processed", "sku_decisioning_grid.csv"))
    print("=== SKU DECISIONING GRID BREAKDOWN ===")
    print(grid["quadrant"].value_counts())
    print(f"\nTotal Sales at Risk: INR {grid['sales_at_risk_inr'].sum():,.2f}")
    print(f"Total Capital Locked: INR {grid['capital_locked_inr'].sum():,.2f}")

if __name__ == "__main__":
    run_risk_inspection()
