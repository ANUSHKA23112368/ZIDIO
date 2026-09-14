"""
Production Scoring Service for NorthBay Living (Project FORESIGHT)
Fulfills Deliverable D6 Acceptance Criteria:
1. Production REST API built with FastAPI and Uvicorn.
2. Serves SKU-level demand forecasts and risk classifications.
3. Supports single SKU lookup, batch scoring, and custom what-if simulation.
4. Robust input validation with Pydantic models.
5. Graceful error handling (404 for unknown SKUs, 422 for invalid payloads).
"""

import os
import logging
from typing import List, Optional
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(
    title="NorthBay Living — FORESIGHT Demand & Risk Scoring Service",
    description="Operational REST API serving weekly SKU-level demand forecasts and inventory risk decisions.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Data Cache
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID_PATH = os.path.join(BASE_DIR, "data", "processed", "sku_decisioning_grid.csv")
FORECAST_PATH = os.path.join(BASE_DIR, "data", "processed", "forward_demand_forecast.csv")

grid_df: Optional[pd.DataFrame] = None
forecast_df: Optional[pd.DataFrame] = None

@app.on_event("startup")
def load_artifacts():
    global grid_df, forecast_df
    try:
        if os.path.exists(GRID_PATH) and os.path.exists(FORECAST_PATH):
            grid_df = pd.read_csv(GRID_PATH)
            forecast_df = pd.read_csv(FORECAST_PATH)
            logging.info(f"Loaded {len(grid_df)} SKUs into scoring service memory.")
        else:
            logging.warning("Processed data files not found! Endpoints will return 503 until generated.")
    except Exception as e:
        logging.error(f"Error loading scoring data: {e}")

# Pydantic Schemas
class HealthResponse(BaseModel):
    status: str = "healthy"
    model_version: str = "lightgbm-v1.0"
    total_skus_loaded: int
    forecast_horizon_weeks: int = 8

class WeeklyForecastPoint(BaseModel):
    forecast_week: str
    week_horizon: int
    predicted_units: float
    lower_ci_80: float
    upper_ci_80: float
    baseline_units: float

class SkuScoreResponse(BaseModel):
    sku_id: str
    category: str
    subcategory: str
    unit_cost: float
    list_price: float
    on_hand_units: int
    on_order_units: int
    lead_time_days: int
    weeks_of_supply: float
    stockout_risk_score: float
    overstock_risk_score: float
    quadrant: str
    quadrant_meaning: str
    recommended_action: str
    recommended_reorder_units: int
    sales_at_risk_inr: float
    capital_locked_inr: float
    weekly_forecast: List[WeeklyForecastPoint]

class BatchScoreRequest(BaseModel):
    sku_ids: List[str] = Field(..., example=["NB-FUR-001", "NB-SMA-051", "NB-DEC-101"])

class SimulationRequest(BaseModel):
    sku_id: str = Field(..., example="NB-FUR-001")
    custom_on_hand: int = Field(..., ge=0, example=15)
    custom_on_order: int = Field(default=0, ge=0, example=0)
    custom_lead_time_days: int = Field(default=21, ge=1, le=120, example=21)
    target_service_level: float = Field(default=0.95, ge=0.80, le=0.99, example=0.95)

# Endpoints
@app.get("/health", response_model=HealthResponse)
def health_check():
    """Service health check and model metadata."""
    if grid_df is None:
        raise HTTPException(status_code=503, detail="Scoring service data not initialized.")
    return HealthResponse(
        status="healthy",
        model_version="lightgbm-v1.0",
        total_skus_loaded=len(grid_df),
        forecast_horizon_weeks=8
    )

@app.get("/skus", response_model=List[str])
def list_skus(category: Optional[str] = None):
    """Retrieve list of active SKU identifiers."""
    if grid_df is None:
        raise HTTPException(status_code=503, detail="Scoring service data not initialized.")
    if category:
        filtered = grid_df[grid_df["category"].str.lower() == category.lower()]
        return filtered["sku_id"].tolist()
    return grid_df["sku_id"].tolist()

@app.get("/score/{sku_id}", response_model=SkuScoreResponse)
def score_sku(sku_id: str):
    """
    Returns demand forecast trajectory and operational risk scoring for a given SKU.
    """
    if grid_df is None or forecast_df is None:
        raise HTTPException(status_code=503, detail="Scoring service data not initialized.")
    
    sku_clean = sku_id.strip().upper()
    row = grid_df[grid_df["sku_id"] == sku_clean]
    
    if row.empty:
        raise HTTPException(
            status_code=404, 
            detail=f"SKU '{sku_id}' not found in active catalog. Valid example: 'NB-FUR-001'"
        )
    
    sku_data = row.iloc[0]
    
    # Get 8-week forecast series
    sku_fc = forecast_df[forecast_df["sku_id"] == sku_clean]
    fc_points = [
        WeeklyForecastPoint(
            forecast_week=r["forecast_week"],
            week_horizon=int(r["week_horizon"]),
            predicted_units=float(r["predicted_units"]),
            lower_ci_80=float(r["lower_ci_80"]),
            upper_ci_80=float(r["upper_ci_80"]),
            baseline_units=float(r["baseline_units"])
        )
        for _, r in sku_fc.iterrows()
    ]
    
    return SkuScoreResponse(
        sku_id=sku_data["sku_id"],
        category=sku_data["category"],
        subcategory=sku_data["subcategory"],
        unit_cost=float(sku_data["unit_cost"]),
        list_price=float(sku_data["list_price"]),
        on_hand_units=int(sku_data["on_hand_units"]),
        on_order_units=int(sku_data["on_order_units"]),
        lead_time_days=int(sku_data["lead_time_days"]),
        weeks_of_supply=float(sku_data["weeks_of_supply"]),
        stockout_risk_score=float(sku_data["stockout_risk_score"]),
        overstock_risk_score=float(sku_data["overstock_risk_score"]),
        quadrant=sku_data["quadrant"],
        quadrant_meaning=sku_data["quadrant_meaning"],
        recommended_action=sku_data["recommended_action"],
        recommended_reorder_units=int(sku_data["recommended_reorder_units"]),
        sales_at_risk_inr=float(sku_data["sales_at_risk_inr"]),
        capital_locked_inr=float(sku_data["capital_locked_inr"]),
        weekly_forecast=fc_points
    )

@app.post("/score/batch", response_model=List[SkuScoreResponse])
def batch_score(payload: BatchScoreRequest):
    """
    Batch scoring endpoint for inventory planners.
    """
    results = []
    for s_id in payload.sku_ids:
        try:
            res = score_sku(s_id)
            results.append(res)
        except HTTPException:
            continue # skip uncatalogued SKUs in batch
    return results

@app.post("/simulate")
def simulate_inventory_risk(req: SimulationRequest):
    """
    What-If Simulator: Evaluate stockout and overstock risk under arbitrary inventory parameters.
    """
    if grid_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded.")
        
    sku_clean = req.sku_id.strip().upper()
    row = grid_df[grid_df["sku_id"] == sku_clean]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"SKU '{req.sku_id}' not found.")
        
    sku_data = row.iloc[0]
    avg_demand = float(sku_data["avg_weekly_demand"])
    std_demand = float(sku_data["demand_std_weekly"])
    unit_cost = float(sku_data["unit_cost"])
    list_price = float(sku_data["list_price"])
    
    # Recalculate lead time metrics
    lead_time_weeks = req.custom_lead_time_days / 7.0
    lt_demand = avg_demand * lead_time_weeks
    z_val = 1.645 if req.target_service_level >= 0.95 else 1.28
    safety_stock = z_val * np.sqrt(lead_time_weeks) * (std_demand + 1.0)
    dynamic_rop = lt_demand + safety_stock
    
    available = req.custom_on_hand + req.custom_on_order
    deficit = max(0.0, lt_demand - available)
    stockout_score = round(min(100.0, max(0.0, ((dynamic_rop - available) / (dynamic_rop + 1e-5)) * 100)), 1)
    sales_at_risk = round(deficit * list_price, 2)
    
    # Overstock metrics
    wos = round(req.custom_on_hand / avg_demand, 1) if avg_demand > 0 else 999.0
    excess = max(0.0, req.custom_on_hand - (avg_demand * 8.0))
    overstock_score = round(min(100.0, max(0.0, ((wos - 8.0) / 16.0) * 100)), 1)
    capital_locked = round(excess * unit_cost, 2)
    
    # Quadrant
    is_high_stockout = stockout_score >= 45.0 or deficit > 0
    is_high_overstock = overstock_score >= 40.0 or wos >= 14.0
    
    if is_high_stockout and not is_high_overstock:
        quadrant = "Reorder now"
        action = "Raise a replenishment order before stock runs out."
    elif is_high_overstock and not is_high_stockout:
        quadrant = "Markdown / clear"
        action = "Promote or discount to free up locked working capital."
    elif is_high_stockout and is_high_overstock:
        quadrant = "Watch / volatile"
        action = "Investigate — erratic demand profile; review supplier lead times manually."
    else:
        quadrant = "Healthy"
        action = "Balanced inventory; no immediate action required."
        
    return {
        "sku_id": sku_clean,
        "simulated_inputs": {
            "on_hand": req.custom_on_hand,
            "on_order": req.custom_on_order,
            "lead_time_days": req.custom_lead_time_days,
            "target_service_level": req.target_service_level
        },
        "metrics": {
            "lead_time_demand": round(lt_demand, 1),
            "safety_stock": round(safety_stock, 1),
            "weeks_of_supply": wos,
            "stockout_risk_score": stockout_score,
            "overstock_risk_score": overstock_score,
            "sales_at_risk_inr": sales_at_risk,
            "capital_locked_inr": capital_locked,
            "quadrant": quadrant,
            "recommended_action": action
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
