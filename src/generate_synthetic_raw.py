"""
Synthetic Raw Data Generator for NorthBay Living (Project FORESIGHT)
Simulates the 4 client extracts specified in Section 05 & Appendix A of the engagement brief:
1. sales_daily.csv (Fact table)
2. sku_master.csv (SKU dimension)
3. calendar.csv (Calendar & promotional dimension)
4. inventory_snapshots.csv (Stock position & lead times)

Includes realistic client imperfections: missing values, slight duplicates,
inconsistent category casing, returns (negatives), and price variations.
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_northbay_data(output_dir="data/raw", seed=42):
    np.random.seed(seed)
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating NorthBay Living synthetic client data in '{output_dir}'...")

    # -------------------------------------------------------------
    # 1. SKU MASTER (~200 active SKUs)
    # -------------------------------------------------------------
    categories = {
        "Furnishings": ["Seating", "Tables", "Shelving", "Rugs", "Bedding"],
        "Décor": ["Lighting", "Wall Art", "Vases & Planters", "Mirrors", "Candles"],
        "Small Appliances": ["Coffee Makers", "Air Fryers", "Blenders", "Toasters", "Electric Kettles"],
        "Kitchenware": ["Cookware Sets", "Cutlery", "Dinnerware", "Storage Containers", "Baking Tools"]
    }
    
    sku_rows = []
    sku_counter = 1
    
    for cat, subcats in categories.items():
        # ~50 SKUs per category
        for _ in range(50):
            subcat = random.choice(subcats)
            sku_id = f"NB-{cat[:3].upper()}-{sku_counter:03d}"
            sku_counter += 1
            
            # Unit economics in INR (₹)
            if cat == "Furnishings":
                unit_cost = round(random.uniform(2500, 18000), 2)
                margin = random.uniform(1.6, 2.4)
            elif cat == "Small Appliances":
                unit_cost = round(random.uniform(1800, 12000), 2)
                margin = random.uniform(1.5, 2.2)
            elif cat == "Décor":
                unit_cost = round(random.uniform(400, 3500), 2)
                margin = random.uniform(1.8, 2.8)
            else: # Kitchenware
                unit_cost = round(random.uniform(300, 4500), 2)
                margin = random.uniform(1.7, 2.5)
                
            list_price = round(unit_cost * margin, 2)
            
            # Launch date between 2023-01-01 and 2024-06-01
            days_ago = random.randint(300, 800)
            launch_date = (datetime(2025, 12, 31) - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            
            # Add deliberate messy casing / whitespace for ~5% of rows
            clean_cat = cat
            if random.random() < 0.05:
                clean_cat = random.choice([cat.lower(), cat.upper(), f"  {cat} "])
                
            sku_rows.append({
                "sku_id": sku_id,
                "category": clean_cat,
                "subcategory": subcat,
                "launch_date": launch_date,
                "unit_cost": unit_cost,
                "list_price": list_price
            })
            
    df_sku = pd.DataFrame(sku_rows)
    # Inject a few missing cost/price values (to test pipeline imputation)
    missing_cost_idx = np.random.choice(df_sku.index, size=3, replace=False)
    df_sku.loc[missing_cost_idx, "unit_cost"] = np.nan
    
    sku_master_path = os.path.join(output_dir, "sku_master.csv")
    df_sku.to_csv(sku_master_path, index=False)
    print(f"-> Saved {len(df_sku)} SKUs to {sku_master_path}")

    # -------------------------------------------------------------
    # 2. CALENDAR (2024-01-01 to 2026-03-31: 821 days = ~117 weeks)
    # -------------------------------------------------------------
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2026, 3, 31)
    date_list = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    
    calendar_rows = []
    
    # Festive events definition
    promo_events_dict = {
        (1, 26): "Republic Day Sale",
        (3, 25): "Holi Spring Festival",
        (8, 15): "Independence Day Super Sale",
        (10, 20): "Diwali Festive Bonanza",
        (10, 21): "Diwali Festive Bonanza",
        (10, 22): "Diwali Festive Bonanza",
        (10, 23): "Diwali Festive Bonanza",
        (11, 28): "Black Friday & Cyber Week",
        (11, 29): "Black Friday & Cyber Week",
        (12, 24): "Year End & Christmas Carnival",
        (12, 25): "Year End & Christmas Carnival",
        (12, 31): "New Year Countdown Sale"
    }

    for dt in date_list:
        m = dt.month
        d = dt.day
        
        # Indian seasons
        if m in [12, 1, 2]:
            season = "Winter"
        elif m in [3, 4, 5]:
            season = "Summer"
        elif m in [6, 7, 8, 9]:
            season = "Monsoon"
        else:
            season = "Festive"
            
        event_name = promo_events_dict.get((m, d), "None")
        is_holiday = 1 if event_name != "None" or dt.weekday() in [5, 6] else 0
        
        calendar_rows.append({
            "date": dt.strftime("%Y-%m-%d"),
            "week": int(dt.strftime("%W")),
            "month": m,
            "season": season,
            "is_holiday": is_holiday,
            "promo_event": None if event_name == "None" else event_name
        })
        
    df_cal = pd.DataFrame(calendar_rows)
    calendar_path = os.path.join(output_dir, "calendar.csv")
    df_cal.to_csv(calendar_path, index=False)
    print(f"-> Saved {len(df_cal)} calendar dates to {calendar_path}")

    # -------------------------------------------------------------
    # 3. SALES DAILY (Fact Table)
    # -------------------------------------------------------------
    # Assign baseline sales velocity per SKU (Pareto 80/20 distribution)
    # Some best-sellers (velocity 15-40 units/day), medium (3-12 units/day), slow/dead movers (0-2 units/day)
    num_skus = len(df_sku)
    sku_ids = df_sku["sku_id"].tolist()
    
    # Base velocity distribution
    sku_base_velocity = {}
    sku_velocity_class = {}
    for i, s_id in enumerate(sku_ids):
        p = i / num_skus
        if p < 0.15: # Top bestsellers (15%)
            vel = np.random.uniform(15, 45)
            cls = "High"
        elif p < 0.65: # Core catalog (50%)
            vel = np.random.uniform(4, 14)
            cls = "Medium"
        elif p < 0.85: # Slow movers (20%)
            vel = np.random.uniform(0.5, 3.5)
            cls = "Slow"
        else: # Dead stock / tail (15%)
            vel = np.random.uniform(0.05, 0.6)
            cls = "Dead"
        sku_base_velocity[s_id] = vel
        sku_velocity_class[s_id] = cls
        
    price_map = df_sku.set_index("sku_id")["list_price"].to_dict()
    
    sales_rows = []
    
    # We sample daily sales with intermittent demand for slow movers
    # To keep file size manageable and training speedy, generate across all SKUs
    for dt in date_list:
        d_str = dt.strftime("%Y-%m-%d")
        cal_row = promo_events_dict.get((dt.month, dt.day), None)
        is_promo_day = 1 if cal_row is not None else (1 if random.random() < 0.08 else 0)
        
        # Weekend lift
        weekend_mult = 1.35 if dt.weekday() in [5, 6] else 1.0
        # Promo event lift
        promo_mult = 2.2 if is_promo_day else 1.0
        # Seasonality lift (Festive season boost in Oct/Nov)
        festive_mult = 1.6 if dt.month in [10, 11] else 1.0
        
        for s_id in sku_ids:
            base_v = sku_base_velocity[s_id]
            expected_units = base_v * weekend_mult * promo_mult * festive_mult
            
            # Poisson or Negative Binomial stochastic realization
            units = np.random.poisson(expected_units)
            
            # If units == 0 and not bestseller, with some probability omit row (real D2C sparse transaction log)
            if units == 0 and sku_velocity_class[s_id] in ["Slow", "Dead"] and random.random() < 0.6:
                continue
                
            list_p = price_map.get(s_id, 1200.0)
            if pd.isna(list_p):
                list_p = 1500.0
                
            unit_discount = random.uniform(0.1, 0.25) if is_promo_day else 0.0
            actual_price = round(list_p * (1 - unit_discount), 2)
            revenue = round(units * actual_price, 2)
            
            sales_rows.append({
                "date": d_str,
                "sku_id": s_id,
                "units_sold": units,
                "revenue": revenue,
                "unit_price": actual_price,
                "promo_flag": is_promo_day
            })
            
    df_sales = pd.DataFrame(sales_rows)
    
    # Inject client data imperfections:
    # A. 0.3% duplicate entries
    n_dupes = int(len(df_sales) * 0.003)
    dupes = df_sales.sample(n=n_dupes, random_state=seed)
    df_sales = pd.concat([df_sales, dupes], ignore_index=True)
    
    # B. A few negative units (returns/order cancellations)
    neg_idx = np.random.choice(df_sales.index, size=40, replace=False)
    df_sales.loc[neg_idx, "units_sold"] = -1 * np.random.randint(1, 4, size=40)
    df_sales.loc[neg_idx, "revenue"] = df_sales.loc[neg_idx, "units_sold"] * df_sales.loc[neg_idx, "unit_price"]
    
    # C. Missing promo_flag for 0.5% rows
    missing_promo_idx = np.random.choice(df_sales.index, size=int(len(df_sales)*0.005), replace=False)
    df_sales.loc[missing_promo_idx, "promo_flag"] = np.nan
    
    # Shuffle slightly
    df_sales = df_sales.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    sales_daily_path = os.path.join(output_dir, "sales_daily.csv")
    df_sales.to_csv(sales_daily_path, index=False)
    print(f"-> Saved {len(df_sales)} daily transaction records to {sales_daily_path}")

    # -------------------------------------------------------------
    # 4. INVENTORY SNAPSHOTS
    # -------------------------------------------------------------
    # Periodic stock position per SKU (e.g. latest snapshot as of 2026-03-31)
    # Designed with NorthBay's exact business dilemma:
    # - Some best-sellers have depleted stock (stockout risk)
    # - Some slow movers have high on-hand (overstock / capital locked)
    inv_rows = []
    snapshot_date = "2026-03-31"
    
    for s_id in sku_ids:
        base_v = sku_base_velocity[s_id]
        v_class = sku_velocity_class[s_id]
        lead_time = random.choice([14, 21, 28, 35, 42]) # 2 to 6 weeks
        
        # Expected demand over lead time (in days)
        lt_demand = base_v * lead_time
        reorder_point = int(lt_demand * 1.35) # safety stock buffer 35%
        
        if v_class == "High":
            # 40% chance of critical stockout risk (stock < lead time demand)
            if random.random() < 0.40:
                on_hand = int(random.uniform(0.1, 0.5) * lt_demand)
                on_order = int(random.uniform(0, 0.3) * lt_demand) # supplier hasn't arrived
            else:
                on_hand = int(random.uniform(1.2, 2.5) * lt_demand)
                on_order = int(random.uniform(0.5, 1.5) * lt_demand)
        elif v_class == "Dead" or v_class == "Slow":
            # 60% chance of extreme overstock (holding 4-12 months of demand)
            if random.random() < 0.60:
                on_hand = int(random.uniform(120, 360) * max(base_v, 0.2))
                on_order = 0
            else:
                on_hand = int(random.uniform(20, 60) * max(base_v, 0.2))
                on_order = 0
        else: # Medium
            on_hand = int(random.uniform(0.8, 2.2) * lt_demand)
            on_order = int(random.uniform(0.0, 1.0) * lt_demand)
            
        inv_rows.append({
            "date": snapshot_date,
            "sku_id": s_id,
            "on_hand_units": max(0, on_hand),
            "on_order_units": max(0, on_order),
            "lead_time_days": lead_time,
            "reorder_point": max(5, reorder_point)
        })
        
    df_inv = pd.DataFrame(inv_rows)
    inventory_path = os.path.join(output_dir, "inventory_snapshots.csv")
    df_inv.to_csv(inventory_path, index=False)
    print(f"-> Saved {len(df_inv)} inventory snapshots to {inventory_path}")
    print("Synthetic raw data generation complete successfully!")

if __name__ == "__main__":
    generate_northbay_data()
