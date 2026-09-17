"""
Project FORESIGHT — Master Pipeline Orchestrator
Executes the full data science workflow end-to-end with a single command:
1. Generates synthetic raw client extracts (if missing)
2. Runs data ingestion, cleaning, and weekly aggregation
3. Performs rolling-origin backtesting & trains LightGBM forecast models
4. Computes stockout and overstock risk and populates decisioning grid
"""

import os
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    start_time = time.time()
    logging.info("==================================================================")
    logging.info("Starting Project FORESIGHT End-to-End Analytics Pipeline")
    logging.info("Client: NorthBay Living • Partner: Zidio Development")
    logging.info("==================================================================")

    # 1. Ensure Raw Data
    raw_sales = os.path.join("data", "raw", "sales_daily.csv")
    if not os.path.exists(raw_sales):
        logging.info("[Step 1/4] Generating raw client extracts...")
        from src.generate_synthetic_raw import generate_northbay_data
        generate_northbay_data()
    else:
        logging.info("[Step 1/4] Raw client extracts already present in data/raw/.")

    # 2. Ingest, Clean & Aggregate
    logging.info("[Step 2/4] Ingesting and cleaning raw extracts (Deliverable D1)...")
    # from src.pipeline import NorthBayDataPipeline
    # pipeline = NorthBayDataPipeline()
    # pipeline.run_pipeline()
    from src.pipeline import run_pipeline as custom_data_pipeline
    custom_data_pipeline()  # <--- Ensure this is indented!


    # 3. Model Training & Rolling-Origin Backtesting
    logging.info("[Step 3/4] Running rolling-origin CV backtest & training ML forecaster (Deliverable D3)...")
    from src.forecast import DemandForecastModel
    forecaster = DemandForecastModel(horizon_weeks=8)
    df_features = forecaster.load_and_engineer_features()
    backtest_results = forecaster.run_rolling_origin_backtest(df_features, n_splits=3)
    forecaster.train_production_models(df_features)
    forecaster.save_model()
    forecast_df = forecaster.generate_forward_forecast(df_features, horizon_weeks=8)
    forecast_df.to_csv(os.path.join("data", "processed", "forward_demand_forecast.csv"), index=False)

    # 4. Risk Scoring & Decisioning Grid
    logging.info("[Step 4/4] Computing stockout & overstock risk scores and rupee values (Deliverable D4)...")
    from src.risk import RiskScoringEngine
    risk_engine = RiskScoringEngine()
    grid_df, summary = risk_engine.calculate_risk(target_service_level=0.95)

    elapsed = time.time() - start_time
    logging.info("==================================================================")
    logging.info(f"Pipeline finished successfully in {elapsed:.2f} seconds!")
    logging.info(f"Forecast Horizon: 8 weeks across {len(grid_df)} active SKUs")
    logging.info(f"WAPE vs Baseline: {backtest_results['avg_model_wape']}% vs {backtest_results['avg_baseline_wape']}% ({backtest_results['avg_wape_reduction']:+.2f}%)")
    logging.info(f"Total Sales at Risk:  ₹{summary['total_sales_at_risk_inr']:,.2f}")
    logging.info(f"Total Locked Capital: ₹{summary['total_capital_locked_inr']:,.2f}")
    logging.info("==================================================================")
    logging.info("To launch the Streamlit Planning Dashboard:  run_dashboard.bat")
    logging.info("To launch the FastAPI Scoring Service:       run_service.bat")
    logging.info("==================================================================")

if __name__ == "__main__":
    main()
