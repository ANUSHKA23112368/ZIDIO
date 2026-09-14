"""
Risk Scoring & Decisioning Engine for NorthBay Living (Project FORESIGHT)
Fulfills Deliverable D4 Acceptance Criteria:
1. Scores stockout risk and overstock risk for every SKU over the forecast horizon.
2. Attaches actionable operational recommendations:
   - "Reorder now" (High stockout, low overstock)
   - "Markdown / clear" (High overstock, low stockout)
   - "Watch / volatile" (High on both)
   - "Healthy" (Low on both)
3. Quantifies financial impact in Rupees (INR):
   - Sales-at-risk (₹) from projected stockouts over lead time.
   - Working capital locked (₹) in excess overstock.
4. Outputs the comprehensive SKU Decisioning Grid for dashboard and service consumption.
"""

import os
import json
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class RiskScoringEngine:
    def __init__(
        self,
        forecast_path="data/processed/forward_demand_forecast.csv",
        inventory_path="data/processed/latest_inventory.csv",
        output_dir="data/processed",
        reports_dir="reports"
    ):
        self.forecast_path = forecast_path
        self.inventory_path = inventory_path
        self.output_dir = output_dir
        self.reports_dir = reports_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

    def calculate_risk(self, target_service_level=0.95):
        logging.info("Calculating stockout and overstock risk scores across all active SKUs...")
        
        forecast_df = pd.read_csv(self.forecast_path)
        inv_df = pd.read_csv(self.inventory_path)
        
        # 1. Aggregate forward forecast per SKU
        # Total 8-week forecast, weekly average, upper bound (p90)
        agg_forecast = forecast_df.groupby("sku_id").agg(
            category=("category", "first"),
            subcategory=("subcategory", "first"),
            unit_cost=("unit_cost", "first"),
            list_price=("list_price", "first"),
            total_8w_forecast=("predicted_units", "sum"),
            avg_weekly_demand=("predicted_units", "mean"),
            demand_std_weekly=("predicted_units", lambda x: float(np.std(x))),
            upper_8w_forecast=("upper_ci_80", "sum"),
            lower_8w_forecast=("lower_ci_80", "sum")
        ).reset_index()
        
        # 2. Merge with current inventory snapshots
        merged = pd.merge(agg_forecast, inv_df, on="sku_id", how="inner")
        
        # 3. Lead Time Demand Calculation
        # lead_time_weeks = lead_time_days / 7.0
        merged["lead_time_weeks"] = (merged["lead_time_days"] / 7.0).round(1)
        merged["lead_time_demand"] = (merged["avg_weekly_demand"] * merged["lead_time_weeks"]).round(1)
        
        # Safety Stock based on target service level (Z ~ 1.645 for 95%) and lead time uncertainty
        z_score = 1.645 if target_service_level >= 0.95 else 1.28
        merged["safety_stock"] = (z_score * np.sqrt(merged["lead_time_weeks"]) * (merged["demand_std_weekly"] + 1.0)).round(1)
        merged["dynamic_reorder_point"] = (merged["lead_time_demand"] + merged["safety_stock"]).round(1)
        
        # Available stock to cover lead time
        merged["available_stock"] = merged["on_hand_units"] + merged["on_order_units"]
        
        # -------------------------------------------------------------
        # 4. Stockout Risk Scoring
        # -------------------------------------------------------------
        # Deficit over lead time
        merged["projected_stockout_units"] = np.maximum(
            0,
            merged["lead_time_demand"] - merged["available_stock"]
        ).round(1)
        
        # Stockout risk score (0 to 100):
        # 100 if stockout is imminent, 0 if ample coverage
        merged["stockout_risk_score"] = np.where(
            merged["lead_time_demand"] <= 0,
            0.0,
            np.clip(
                ((merged["dynamic_reorder_point"] - merged["available_stock"]) / (merged["dynamic_reorder_point"] + 1e-5)) * 100,
                0.0,
                100.0
            )
        ).round(1)
        
        # Sales at risk (₹ INR) = projected_stockout_units * list_price
        merged["sales_at_risk_inr"] = (merged["projected_stockout_units"] * merged["list_price"]).round(2)
        
        # -------------------------------------------------------------
        # 5. Overstock Risk Scoring
        # -------------------------------------------------------------
        # Weeks of Supply (WOS) = on_hand_units / avg_weekly_demand
        merged["weeks_of_supply"] = np.where(
            merged["avg_weekly_demand"] > 0,
            (merged["on_hand_units"] / merged["avg_weekly_demand"]).round(1),
            999.0
        )
        
        # Ideal forward holding is 6-8 weeks
        # Overstock excess = on_hand_units - (8 weeks demand)
        merged["excess_units"] = np.maximum(
            0,
            merged["on_hand_units"] - (merged["avg_weekly_demand"] * 8.0)
        ).round(1)
        
        # Overstock risk score (0 to 100)
        # 0 if WOS <= 8 weeks, scaled up to 100 if WOS >= 24 weeks
        merged["overstock_risk_score"] = np.clip(
            ((merged["weeks_of_supply"] - 8.0) / 16.0) * 100.0,
            0.0,
            100.0
        ).round(1)
        
        # Capital locked (₹ INR) = excess_units * unit_cost
        merged["capital_locked_inr"] = (merged["excess_units"] * merged["unit_cost"]).round(2)

        # -------------------------------------------------------------
        # 6. Decisioning Grid Classification (Section 08.2)
        # -------------------------------------------------------------
        def assign_decision(row):
            is_high_stockout = row["stockout_risk_score"] >= 45.0 or row["projected_stockout_units"] > 0
            is_high_overstock = row["overstock_risk_score"] >= 40.0 or row["weeks_of_supply"] >= 14.0
            
            if is_high_stockout and not is_high_overstock:
                return "Reorder now", "High stockout, low overstock", "Raise a replenishment order before stock runs out."
            elif is_high_overstock and not is_high_stockout:
                return "Markdown / clear", "High overstock, low stockout", "Promote or discount to free up locked working capital."
            elif is_high_stockout and is_high_overstock:
                return "Watch / volatile", "High on both", "Investigate — erratic demand profile; review supplier lead times manually."
            else:
                return "Healthy", "Low on both", "Balanced inventory; no immediate action required."

        decisions = merged.apply(assign_decision, axis=1)
        merged["quadrant"] = [d[0] for d in decisions]
        merged["quadrant_meaning"] = [d[1] for d in decisions]
        merged["recommended_action"] = [d[2] for d in decisions]
        
        # Recommended reorder quantity when "Reorder now"
        # Bring stock up to 8 weeks demand + safety stock
        merged["recommended_reorder_units"] = np.where(
            merged["quadrant"] == "Reorder now",
            np.maximum(0, (merged["avg_weekly_demand"] * 8.0 + merged["safety_stock"] - merged["available_stock"]).round(0)),
            0
        ).astype(int)

        # 7. Summary Metrics
        total_sales_at_risk = float(merged["sales_at_risk_inr"].sum())
        total_capital_locked = float(merged["capital_locked_inr"].sum())
        quadrant_counts = merged["quadrant"].value_counts().to_dict()
        
        summary = {
            "total_skus": len(merged),
            "total_sales_at_risk_inr": total_sales_at_risk,
            "total_capital_locked_inr": total_capital_locked,
            "quadrant_distribution": quadrant_counts,
            "top_reorder_skus": merged[merged["quadrant"] == "Reorder now"].sort_values("sales_at_risk_inr", ascending=False).head(10)[["sku_id", "category", "sales_at_risk_inr", "recommended_reorder_units"]].to_dict(orient="records"),
            "top_markdown_skus": merged[merged["quadrant"] == "Markdown / clear"].sort_values("capital_locked_inr", ascending=False).head(10)[["sku_id", "category", "capital_locked_inr", "excess_units", "weeks_of_supply"]].to_dict(orient="records")
        }
        
        logging.info("=== RISK SUMMARY ===")
        logging.info(f"Total Sales at Risk:      ₹{total_sales_at_risk:,.2f}")
        logging.info(f"Total Capital Locked:     ₹{total_capital_locked:,.2f}")
        logging.info(f"Quadrant Breakdown:       {quadrant_counts}")
        
        # Save output
        grid_path = os.path.join(self.output_dir, "sku_decisioning_grid.csv")
        merged.to_csv(grid_path, index=False)
        
        summary_path = os.path.join(self.reports_dir, "risk_executive_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
            
        logging.info(f"Decisioning grid exported to {grid_path}")
        return merged, summary

if __name__ == "__main__":
    engine = RiskScoringEngine()
    grid_df, summary = engine.calculate_risk()
