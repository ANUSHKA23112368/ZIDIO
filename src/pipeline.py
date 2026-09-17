import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import numpy as np



# 1. Custom Transformer for Automated Cleaning & Imputation
# 1. Custom Transformer for Automated Cleaning & Imputation
# 1. Custom Transformer for Automated Cleaning & Imputation
# 1. Custom Transformer for Automated Cleaning & Imputation
class CalendarPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self 
        
    def transform(self, X):
        df = X.copy()
        
        # Standardize column names
        df.columns = df.columns.str.lower().str.strip()
        print(f"\n[DEBUG] Calendar columns detected: {df.columns.tolist()}")
        
        # Standardize date format
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
            # --- THE FIX: Auto-generate missing temporal columns from the date ---
            if 'day_of_week' not in df.columns:
                df['day_of_week'] = df['date'].dt.day_name()
            if 'month' not in df.columns:
                df['month'] = df['date'].dt.month
            if 'week' not in df.columns:
                df['week'] = df['date'].dt.isocalendar().week
                
        # Safely handle 'holiday'
        if 'holiday' in df.columns:
            df['holiday'] = df['holiday'].fillna('No Holiday').astype('category')
        else:
            df['holiday'] = 'No Holiday'
            df['holiday'] = df['holiday'].astype('category')
            
        # Safely handle 'promotion_event'
        if 'promotion_event' in df.columns:
            df['promotion_event'] = df['promotion_event'].fillna('No Promotion').astype('category')
        elif 'promo_event' in df.columns: # Catch her specific typo from the debug log
            df['promotion_event'] = df['promo_event'].fillna('No Promotion').astype('category')
        else:
            df['promotion_event'] = 'No Promotion'
            df['promotion_event'] = df['promotion_event'].astype('category')
            
        if 'is_holiday' not in df.columns:
            df['is_holiday'] = 0
            
        if 'season' in df.columns:
            df['season'] = df['season'].astype('category')
            
        return df

def execute_eda_and_pipeline(csv_path):
    # 2. Load Raw Data
    raw_df = pd.read_csv(csv_path)
    
    # 3. Define and Run Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', CalendarPreprocessor())
    ])
    
    cleaned_df = pipeline.fit_transform(raw_df)
    
    # 4. Exploratory Data Analysis (EDA)
    print("--- Pipeline Execution Complete ---")
    print(f"Remaining Missing Values:\n{cleaned_df.isnull().sum()}\n")
    
    # Set up insight visualizations for Deliverable 2
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot A: Distribution of Seasons
    sns.countplot(data=cleaned_df, x='season', ax=axes[0], palette='viridis')
    axes[0].set_title('Days per Season (2024-2025)')
    
    # Plot B: Frequency of Promotions
    sns.countplot(data=cleaned_df, x='promotion_event', ax=axes[1], palette='magma')
    axes[1].set_title('Distribution of Promotion Events')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # 5. Export for Model Engineers
    cleaned_df.to_csv("processed_calendar.csv", index=False)
    print("Ready for handoff: 'processed_calendar.csv' saved.")
    
    return cleaned_df

class SalesPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self 
        
    def transform(self, X):
        df = X.copy()
        
        # Defensive formatting & Debugging
        df.columns = df.columns.str.lower().str.strip()
        
        # Expand rename_map to catch the pricing and promotion aliases
        rename_map = {
            'sku_id': 'sku', 
            'product_id': 'sku', 
            'item_id': 'sku',
            'unit_price': 'price',      # <-- Translates her column for your FinancialScaler
            'promo_flag': 'promotion'   # <-- Future-proofing
        }
        df.rename(columns=rename_map, inplace=True)
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        if 'sku' in df.columns:
            df['sku'] = df['sku'].astype('category')
            
        if 'units_sold' in df.columns and 'revenue' in df.columns:
            df = df[(df['units_sold'] >= 0) & (df['revenue'] >= 0)]
            
        return df

def execute_sales_pipeline(csv_path):
    # 2. Load Raw Data
    raw_df = pd.read_csv(csv_path)
    
    # 3. Define and Run Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', SalesPreprocessor())
    ])
    
    cleaned_df = pipeline.fit_transform(raw_df)
    
    # 4. Exploratory Data Analysis (EDA)
    print("--- Sales Pipeline Execution Complete ---")
    print(f"Total valid records processed: {len(cleaned_df)}\n")
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot A: Daily Aggregated Sales Trend over Time
    daily_sales = cleaned_df.groupby('date')['units_sold'].sum().reset_index()
    sns.lineplot(data=daily_sales, x='date', y='units_sold', ax=axes[0], color='teal')
    axes[0].set_title('Total Daily Units Sold (All SKUs)')
    axes[0].set_ylabel('Total Units')
    axes[0].set_xlabel('Date')
    
    # Plot B: Top 15 SKUs by Total Volume
    top_skus = cleaned_df.groupby('sku')['units_sold'].sum().nlargest(15).reset_index()
    sns.barplot(data=top_skus, x='sku', y='units_sold', ax=axes[1], palette='crest')
    axes[1].set_title('Top 15 Bestselling SKUs (Volume)')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_ylabel('Total Units Sold')
    
    plt.tight_layout()
    plt.show()
    
    # 5. Export for Model Engineers
    cleaned_df.to_csv("processed_sales_daily.csv", index=False)
    print("Ready for handoff: 'processed_sales_daily.csv' saved.")
    
    return cleaned_df

# 1. Custom Transformer for SKU Data Validation & Standardization
# --- REPLACE SKUPreprocessor ---
class SKUPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self 
        
    def transform(self, X):
        df = X.copy()
        
        # Defensive formatting & Debugging
        df.columns = df.columns.str.lower().str.strip()
        
        # Catch alias variations generated by synthetic scripts (including prices)
        rename_map = {
            'sku_id': 'sku', 
            'product_id': 'sku', 
            'item_id': 'sku',
            'unit_cost': 'cost_price',
            'list_price': 'selling_price'
        }
        df.rename(columns=rename_map, inplace=True)
        
        # Construct gross margin if it doesn't exist
        if 'gross_margin_per_unit' not in df.columns:
            if 'selling_price' in df.columns and 'cost_price' in df.columns:
                df['gross_margin_per_unit'] = df['selling_price'] - df['cost_price']
            else:
                df['gross_margin_per_unit'] = 0 # Fallback failsafe
        
        if 'launch_date' in df.columns:
            df['launch_date'] = pd.to_datetime(df['launch_date'])
            
        categorical_cols = ['category', 'subcategory']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
                
        return df

def execute_sku_pipeline(csv_path):
    # 2. Load Raw Data
    raw_df = pd.read_csv(csv_path)
    
    # 3. Define and Run Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', SKUPreprocessor())
    ])
    
    cleaned_df = pipeline.fit_transform(raw_df)
    
    # 4. Exploratory Data Analysis (EDA) & Anomaly Flagging
    print("--- SKU Pipeline Execution Complete ---")
    
    # Isolate negative margin SKUs for the Deliverable 2 Insight Memo
    loss_leaders = cleaned_df[cleaned_df['gross_margin_per_unit'] < 0]
    print(f"ALERT: Discovered {len(loss_leaders)} SKUs with negative gross margins.")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot A: Product Distribution by Category
    sns.countplot(
        data=cleaned_df, 
        y='category', 
        ax=axes[0], 
        palette='mako', 
        order=cleaned_df['category'].value_counts().index
    )
    axes[0].set_title('Product Count by Category')
    axes[0].set_xlabel('Number of SKUs')
    
    # Plot B: Gross Margin Profitability Distribution
    sns.histplot(data=cleaned_df, x='gross_margin_per_unit', ax=axes[1], color='crimson', kde=True)
    axes[1].axvline(0, color='black', linestyle='--', linewidth=2) # Line denoting zero profit
    axes[1].set_title('Distribution of Gross Margins')
    axes[1].set_xlabel('Gross Margin Per Unit (Negative = Loss)')
    
    plt.tight_layout()
    plt.show()
    
    # 5. Export for Model Engineers
    cleaned_df.to_csv("processed_sku_masters.csv", index=False)
    print("Ready for handoff: 'processed_sku_masters.csv' saved.")
    
    return cleaned_df

class InventoryPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self 
        
    def transform(self, X):
        df = X.copy()
        
        # Defensive formatting & Debugging
        df.columns = df.columns.str.lower().str.strip()
        print(f"\n[DEBUG] Inventory columns detected: {df.columns.tolist()}")
        
        rename_map = {'sku_id': 'sku', 'product_id': 'sku', 'item_id': 'sku'}
        df.rename(columns=rename_map, inplace=True)
        
        if 'snapshot_date' in df.columns:
            df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
        if 'sku' in df.columns:
            df['sku'] = df['sku'].astype('category')
            
        # Safe Feature Engineering
        if 'current_stock' in df.columns and 'on_order' in df.columns:
            df['effective_inventory'] = df['current_stock'] + df['on_order']
        else:
            df['effective_inventory'] = 0
            
        def flag_inventory_risk(row):
            # Using .get() prevents crashes if the column doesn't exist
            eff = row.get('effective_inventory', 0)
            safe = row.get('safety_stock', 0)
            reorder = row.get('reorder_point', 0)
            
            if eff <= safe:
                return 'Critical Stockout Risk'
            elif eff <= reorder:
                return 'Reorder Required'
            else:
                return 'Healthy Stock'
                
        df['stock_status'] = df.apply(flag_inventory_risk, axis=1).astype('category')
        
        return df

def execute_inventory_pipeline(csv_path):
    # 2. Load Raw Data
    raw_df = pd.read_csv(csv_path)
    
    # 3. Define and Run Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', InventoryPreprocessor())
    ])
    
    cleaned_df = pipeline.fit_transform(raw_df)
    
    # 4. Exploratory Data Analysis (EDA)
    print("--- Inventory Pipeline Execution Complete ---")
    
    # Calculate financial risk for Deliverable 2 Insight
    critical_items = cleaned_df[cleaned_df['stock_status'] == 'Critical Stockout Risk']
    capital_at_risk = critical_items['inventory_value'].sum()
    print(f"Insight: {len(critical_items)} snapshot records show critical stockout risk.")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot A: Overall Health of Inventory Snapshots
    sns.countplot(
        data=cleaned_df, 
        x='stock_status', 
        ax=axes[0], 
        palette=['#2ecc71', '#f1c40f', '#e74c3c'],
        order=['Healthy Stock', 'Reorder Required', 'Critical Stockout Risk']
    )
    axes[0].set_title('Inventory Health Status Frequency')
    axes[0].set_ylabel('Number of Snapshot Records')
    
    # Plot B: Distribution of Capital Locked in Inventory
    sns.histplot(data=cleaned_df, x='inventory_value', ax=axes[1], color='slateBlue', bins=40, kde=True)
    axes[1].set_title('Distribution of Locked Inventory Value')
    axes[1].set_xlabel('Inventory Value (Capital)')
    
    plt.tight_layout()
    plt.show()
    
    # 5. Export for Model Engineers
    cleaned_df.to_csv("processed_inventory_snapshots.csv", index=False)
    print("Ready for handoff: 'processed_inventory_snapshots.csv' saved.")
    
    return cleaned_df

# 1. Feature Construction: Cyclical Temporal Encoding
class CyclicalEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        df = X.copy()
        
        # Map textual days to numeric representation (0-6)
        day_map = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 
            'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }
        df['day_of_week_num'] = df['day_of_week'].map(day_map)
        
        # Apply Sine/Cosine transformations to preserve cyclical distance 
        # (e.g., ensuring December is mathematically close to January)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)
        
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week_num'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week_num'] / 7)
        
        # Drop the temporary numeric column
        df = df.drop(columns=['day_of_week_num'])
        return df

# 2. Feature Construction: Event Proximity Calculations
class EventProximityConstructor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        df = X.copy()
        # Ensure chronological order for rolling calculations
        df = df.sort_values('date').reset_index(drop=True)
        
        # Calculate days until the next holiday
        holiday_idx = df.index[df['is_holiday'] == 1].tolist()
        if holiday_idx:
            df['days_to_next_holiday'] = df.index.map(
                lambda x: min([h - x for h in holiday_idx if h >= x], default=999)
            )
        else:
            df['days_to_next_holiday'] = 999
            
        # Calculate days until the next promotion event
        promo_idx = df.index[df['promotion_event'] != 'No Promotion'].tolist()
        if promo_idx:
            df['days_to_next_promo'] = df.index.map(
                lambda x: min([p - x for p in promo_idx if p >= x], default=999)
            )
        else:
            df['days_to_next_promo'] = 999
            
        return df

def execute_calendar_feature_engineering(csv_path):
    # Load the output from your previous cleaning pipeline
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Run the feature engineering pipeline
    fe_pipeline = Pipeline(steps=[
        ('cyclical_encoding', CyclicalEncoder()),
        ('event_proximity', EventProximityConstructor())
    ])
    
    engineered_df = fe_pipeline.fit_transform(df)
    
    # Export for the next stage
    engineered_df.to_csv("engineered_calendar.csv", index=False)
    print(f"Feature engineering complete. Added {len(engineered_df.columns) - len(df.columns)} new features.")
    
    return engineered_df

# 1. Feature Construction: Time-Series Lags and Rolling Windows
class TimeSeriesConstructor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        df = X.copy()
        
        # MUST sort by SKU and Date to prevent data leakage between different products
        df = df.sort_values(by=['sku', 'date']).reset_index(drop=True)
        
        # Construct Lag Features: Past sales performance
        df['units_sold_lag_1'] = df.groupby('sku')['units_sold'].shift(1)
        df['units_sold_lag_7'] = df.groupby('sku')['units_sold'].shift(7)
        
        # Construct Rolling Features: Momentum and smoothing
        # Using transform to broadcast the rolling calculation back to the original rows
        df['units_sold_rolling_mean_7'] = df.groupby('sku')['units_sold'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        
        # Fill NaN values generated by shifting the first few rows of each SKU
        df.fillna(0, inplace=True)
        
        return df

# 2. Feature Scaling: Normalizing Financial Metrics
class FinancialScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
        
    def fit(self, X, y=None):
        # Fit the scaler only on the revenue and price columns
        self.scaler.fit(X[['revenue', 'price']])
        return self
        
    def transform(self, X):
        df = X.copy()
        
        # Scale continuous financial features to standard normal distribution (mean=0, std=1)
        scaled_features = self.scaler.transform(df[['revenue', 'price']])
        df['revenue_scaled'] = scaled_features[:, 0]
        df['price_scaled'] = scaled_features[:, 1]
        
        # Drop original unscaled financial columns to prevent multicollinearity
        df = df.drop(columns=['revenue', 'price'])
        
        return df

def execute_sales_feature_engineering(csv_path):
    # Load the output from your previous cleaning pipeline
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Run the feature engineering pipeline
    fe_pipeline = Pipeline(steps=[
        ('ts_constructor', TimeSeriesConstructor()),
        ('fin_scaler', FinancialScaler())
    ])
    
    engineered_df = fe_pipeline.fit_transform(df)
    
    # Export for the ML modeling stage
    engineered_df.to_csv("engineered_sales_daily.csv", index=False)
    print(f"Feature engineering complete. Dataset shape: {engineered_df.shape}")
    
    return engineered_df

from sklearn.preprocessing import StandardScaler

# 1. Feature Construction: Product Age and Profitability Flags
class SKUFeatureConstructor(BaseEstimator, TransformerMixin):
    def __init__(self, reference_date='2025-12-31'):
        # Using the end of the forecasting period as the reference date for age
        self.reference_date = pd.to_datetime(reference_date)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        
        # Construct Feature: Product Age in days
        df['product_age_days'] = (self.reference_date - df['launch_date']).dt.days
        
        # Construct Feature: Binary flag for loss-leading products
        df['is_loss_leader'] = (df['gross_margin_per_unit'] < 0).astype(int)
        
        # Drop raw text and datetime columns that models cannot process
        columns_to_drop = ['product_name', 'launch_date']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
            
        return df

# 2. Feature Scaling: Normalizing Financial Metrics
class SKUFinancialScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        # Fit scaler on the specific financial columns
        self.scaler.fit(X[['cost_price', 'selling_price', 'gross_margin_per_unit']])
        return self

    def transform(self, X):
        df = X.copy()
        
        # Scale continuous financial features to standard normal distribution
        scaled_features = self.scaler.transform(df[['cost_price', 'selling_price', 'gross_margin_per_unit']])
        df['cost_price_scaled'] = scaled_features[:, 0]
        df['selling_price_scaled'] = scaled_features[:, 1]
        df['gross_margin_scaled'] = scaled_features[:, 2]
        
        # Drop original unscaled financial columns to prevent multicollinearity
        df = df.drop(columns=['cost_price', 'selling_price', 'gross_margin_per_unit'])
        
        return df

def execute_sku_feature_engineering(csv_path):
    # Load the output from your previous cleaning pipeline
    df = pd.read_csv(csv_path)
    df['launch_date'] = pd.to_datetime(df['launch_date'])
    
    # Run the feature engineering pipeline
    fe_pipeline = Pipeline(steps=[
        ('feature_constructor', SKUFeatureConstructor()),
        ('financial_scaler', SKUFinancialScaler())
    ])
    
    engineered_df = fe_pipeline.fit_transform(df)
    
    # Export for the ML modeling stage
    engineered_df.to_csv("engineered_sku_masters.csv", index=False)
    print(f"Feature engineering complete. Dataset shape: {engineered_df.shape}")
    
    return engineered_df

# 1. Feature Construction: Relative Risk Ratios and Encoding
class InventoryFeatureConstructor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        
        # Construct Feature: Proximity to reorder threshold (lower = higher risk)
        # Adding a tiny epsilon (1e-5) to prevent division by zero errors
        df['stock_to_reorder_ratio'] = df['current_stock'] / (df['reorder_point'] + 1e-5)
        
        # Construct Feature: Absolute buffer before breaching safety stock
        df['safety_buffer_units'] = df['current_stock'] - df['safety_stock']
        
        # Ordinal Encoding: Convert categorical risk text to machine-readable integers
        # This assumes 'stock_status' was generated in the previous cleaning pipeline
        status_map = {
            'Healthy Stock': 0, 
            'Reorder Required': 1, 
            'Critical Stockout Risk': 2
        }
        if 'stock_status' in df.columns:
            df['stock_status_encoded'] = df['stock_status'].map(status_map)
            df = df.drop(columns=['stock_status'])
            
        return df

# 2. Feature Scaling: Normalizing Volumes and Capital
class InventoryScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
        # Define all continuous columns that require standardization
        self.cols_to_scale = [
            'current_stock', 'on_order', 'lead_time_days', 
            'safety_stock', 'reorder_point', 'inventory_value',
            'effective_inventory', 'safety_buffer_units', 'stock_to_reorder_ratio'
        ]

    def fit(self, X, y=None):
        # Identify which columns actually exist in the dataframe to prevent KeyError
        self.existing_cols = [col for col in self.cols_to_scale if col in X.columns]
        self.scaler.fit(X[self.existing_cols])
        return self

    def transform(self, X):
        df = X.copy()
        
        # Apply standard scaling (mean=0, std=1)
        scaled_features = self.scaler.transform(df[self.existing_cols])
        
        # Append scaled columns to the dataframe
        for i, col in enumerate(self.existing_cols):
            df[f'{col}_scaled'] = scaled_features[:, i]
            
        # Drop original unscaled columns to prevent multicollinearity
        df = df.drop(columns=self.existing_cols)
        
        return df

def execute_inventory_feature_engineering(csv_path):
    # Load the output from your previous cleaning pipeline
    df = pd.read_csv(csv_path)
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
    
    # Run the feature engineering pipeline
    fe_pipeline = Pipeline(steps=[
        ('feature_constructor', InventoryFeatureConstructor()),
        ('inventory_scaler', InventoryScaler())
    ])
    
    engineered_df = fe_pipeline.fit_transform(df)
    
    # Export for the ML modeling stage
    engineered_df.to_csv("engineered_inventory_snapshots.csv", index=False)
    print(f"Feature engineering complete. Dataset shape: {engineered_df.shape}")
    
    return engineered_df

def evaluate_baseline_wape(engineered_sales_path):
    # Load your engineered sales data
    df = pd.read_csv(engineered_sales_path)
    
    # Drop rows where lag features are 0 due to the initial shifting (first 7 days of dataset)
    # to ensure a fair evaluation of the baseline
    eval_df = df[df['units_sold_lag_7'] > 0].copy()
    
    # Calculate absolute error between actual sales and the 7-day naive forecast
    eval_df['absolute_error'] = np.abs(eval_df['units_sold'] - eval_df['units_sold_lag_7'])
    
    # Calculate WAPE
    total_absolute_error = eval_df['absolute_error'].sum()
    total_actual_sales = eval_df['units_sold'].sum()
    
    wape_score = total_absolute_error / total_actual_sales
    
    print("--- Baseline Evaluation Complete ---")
    print(f"Total Evaluation Records: {len(eval_df)}")
    print(f"Seasonal-Naive Baseline WAPE: {wape_score:.4%} \n")
    print("Presentation Note: Your ML teammates must build a model that scores LOWER than this WAPE percentage.")
    
    return wape_score

import os

def run_foresight_master_pipeline():
    print("Starting FORESIGHT Master Data Pipeline...\n")
    
    # 1. Execute Cleaning & Trigger EDA Visualizations (Deliverable 2)
    print("--- PHASE 1: Data Cleaning & Insight Generation ---")
    clean_cal = execute_eda_and_pipeline("calendar.csv")
    clean_sales = execute_sales_pipeline("sales_daily.csv")
    clean_sku = execute_sku_pipeline("sku_master.csv")
    clean_inv = execute_inventory_pipeline("inventory_snapshots.csv")
    
    # 2. Execute Feature Engineering (Deliverable 1)
    print("\n--- PHASE 2: Feature Engineering & Scaling ---")
    eng_cal = execute_calendar_feature_engineering("processed_calendar.csv")
    eng_sales = execute_sales_feature_engineering("processed_sales_daily.csv")
    eng_sku = execute_sku_feature_engineering("processed_sku_masters.csv")
    eng_inv = execute_inventory_feature_engineering("processed_inventory_snapshots.csv")
    
    # 3. Calculate Target Metric
    print("\n--- PHASE 3: Baseline Evaluation ---")
    baseline_wape = evaluate_baseline_wape("engineered_sales_daily.csv")
    
    print("\nSUCCESS: All pipelines executed. Handing off to ML Engineering.")


def generate_weekly_ml_features(eng_sales, eng_cal, eng_sku):
    """
    Final Adapter function bridging the engineered data to the LightGBM schema.
    """
    print("--- Merging & Aggregating for ML Handoff ---")
    
    # 1. Merge all datasets
    df = eng_sales.merge(eng_cal, on='date', how='left')
    df = df.merge(eng_sku, on='sku', how='left')
    df['date'] = pd.to_datetime(df['date'])
    
    # 2. Convert text flags to the binary integers her LightGBM model expects
    if 'is_holiday' in df.columns:
        df['holidays_in_week'] = df['is_holiday'].astype(int)
    else:
        df['holidays_in_week'] = 0
        
    promo_col = 'promotion' if 'promotion' in df.columns else 'promotion_event'
    if promo_col in df.columns:
        df['promo_events_in_week'] = (df[promo_col] != 'No Promotion').astype(int)
    else:
        df['promo_events_in_week'] = 0
        
    # 3. Explicitly define math for engineered metrics
    agg_dict = {
        'units_sold': 'sum',
        'units_sold_lag_1': 'sum',
        'units_sold_lag_7': 'sum',
        'units_sold_rolling_mean_7': 'mean',
        'price_scaled': 'mean',
        'holidays_in_week': 'max',
        'promo_events_in_week': 'max'
    }
    
    # 4. Catch-all net for remaining columns
    for col in df.columns:
        if col not in agg_dict and col not in ['sku', 'date']:
            agg_dict[col] = 'first'
            
    weekly_df = df.groupby(['sku', pd.Grouper(key='date', freq='W-SUN')]).agg(agg_dict).reset_index()
    
    # 5. THE FINAL TRANSLATION DICTIONARY
    weekly_df.rename(columns={
        'date': 'week_start', 
        'sku': 'sku_id',
        'week': 'week_of_year',
        'units_sold': 'weekly_units',
        'price_scaled': 'avg_unit_price',
        'selling_price_scaled': 'list_price',
        'cost_price_scaled': 'unit_cost',
        'gross_margin_scaled': 'margin_pct'
    }, inplace=True)
    
    # 6. Save
    import os
    os.makedirs('data/processed', exist_ok=True)
    output_path = 'data/processed/weekly_demand_features.csv'
    weekly_df.to_csv(output_path, index=False)
    print(f"SUCCESS: Exported ML-ready file to {output_path}")

def run_pipeline():
    """Master function called by her run_pipeline.py script"""
    print("Starting Custom Feature Engineering Pipeline...")
    
    # 1. Point to her raw data folder
    raw_dir = "data/raw/"
    
    # 2. Execute your cleaning pipelines
    clean_cal = execute_eda_and_pipeline(f"{raw_dir}calendar.csv")
    clean_sales = execute_sales_pipeline(f"{raw_dir}sales_daily.csv")
    clean_sku = execute_sku_pipeline(f"{raw_dir}sku_master.csv")
    
    # 3. Execute your feature engineering
    # Note: Pass the dataframes directly if you modify your execute functions, 
    # or save them temporarily to data/processed/ and load them back.
    eng_cal = execute_calendar_feature_engineering("processed_calendar.csv")
    eng_sales = execute_sales_feature_engineering("processed_sales_daily.csv")
    eng_sku = execute_sku_feature_engineering("processed_sku_masters.csv")
    
    # 4. Create the final weekly file for her model
    generate_weekly_ml_features(eng_sales, eng_cal, eng_sku)



