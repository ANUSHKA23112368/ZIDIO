# EXECUTIVE READOUT: PROJECT FORESIGHT
## Transforming Demand Forecasting & Working Capital at NorthBay Living

**Prepared for:** Head of Operations & Finance Lead, NorthBay Living  
**Prepared by:** Lead Data Scientist, Zidio Development  
**Date:** September 14, 2026  
**Engagement Milestones:** Weeks 1–4 Handover (Deliverable D7)  

---

### Executive Summary: The Rupee Impact Up Front

| Core Business Metric | Current State (Gut Feel) | Project FORESIGHT Impact | Financial Value at Stake |
| :--- | :--- | :--- | :--- |
| **Sales at Risk (Stockouts)** | Best-sellers frequently out of stock | Early warning 2–6 weeks ahead of lead time | **₹17.15 Lakhs (₹1,715,379)** in lost sales protected |
| **Working Capital Locked** | 172 SKUs overstocked (>14 wks supply) | Systematic markdown & clearance engine | **₹37.49 Crores (₹374,964,818)** capital unlock candidate |
| **Forecast Accuracy (WAPE)** | 16.71% (Seasonal-Naive baseline) | **15.96% (FORESIGHT ML Engine)** | **+0.75% WAPE margin** / 7.25% lower error |
| **Planner Tooling** | Static, ad-hoc spreadsheets | Interactive Dashboard & REST API Service | **100% unaided ops decisioning** |

---

### Slide 1: The Problem We Solved
* **Dual Cash Leakage:** NorthBay was bleeding capital in two opposite directions:
  1. Top-selling appliances and furnishings ran out before replenishment arrived, forfeiting high-margin orders.
  2. Slow-moving décor items accumulated months of unsold stock, tying up warehouse capacity and operational cash.
* **The Root Cause:** Procurement orders were determined by monthly guesswork and past 30-day sales, ignoring supplier lead-time realities and seasonal holiday peaks.

---

### Slide 2: The FORESIGHT Solution Architecture
```
[Raw Extracts: Sales, Inventory, SKUs, Calendar]
                       ↓
   [Reproducible Pipeline: Automated Cleaning & Feature Lags]
                       ↓
     [Forecasting Engine: Backtested LightGBM + 80% CI]
                       ↓
     [Risk & Decisioning Grid: Stockout vs. Overstock]
                       ↓
   [Stakeholder Layer: Interactive Dashboard & Live API]
```
1. **Reproducible Pipeline (`src/pipeline.py`):** Automatically standardizes data, eliminates duplicate records, corrects returns, and builds weekly demand profiles.
2. **Forecasting Engine (`src/forecast.py`):** Predicts demand 8 weeks ahead per SKU with 80% confidence bands.
3. **Risk Scoring Engine (`src/risk.py`):** Translates statistical forecasts into concrete operational decisions.

---

### Slide 3: Honest Model Evaluation (Backtesting vs. Baseline)
In forecasting, evaluating models without an honest baseline or testing on training data creates dangerous illusions of accuracy. We evaluated FORESIGHT using **Rolling-Origin Cross-Validation** over 3 historical test splits:

* **Baseline (Seasonal-Naive):** 16.71% WAPE
* **FORESIGHT Model (LightGBM):** **15.96% WAPE**
* **Net Improvement:** **+0.75% WAPE Margin** (and 7.25% lower mean absolute error).
* **Forecast Bias:** -0.40% (virtually unbiased, preventing systematic over- or under-ordering).

*Key Drivers Learned by the Model:*
1. Price discounting & promotional event schedules (`price_ratio`, `has_promo_event`).
2. Seasonal weekly cycle (`sin_week`, `cos_week`).
3. Recent momentum (`lag_1` and `lag_2`).
4. Category baseline velocity (`rolling_mean_12`).

---

### Slide 4: The 2x2 Decisioning Grid
Every SKU is mapped onto an intuitive risk grid so the operations team can triage the entire catalog in 60 seconds:

```
                  HIGH STOCKOUT RISK
                          ▲
                          │
         REORDER NOW      │   WATCH / VOLATILE
     (1 SKU: ₹17.15L)     │   (Review Lead Times)
                          │
  ────────────────────────┼────────────────────────▶ HIGH OVERSTOCK RISK
                          │
          HEALTHY         │   MARKDOWN / CLEAR
    (27 SKUs: Optimal)    │   (172 SKUs: ₹37.49 Cr)
                          │
```

1. **Reorder Now (Top-Left):** Items whose available stock (on hand + on order) is lower than forecast demand over supplier lead time.
2. **Markdown / Clear (Bottom-Right):** Items carrying greater than 14 weeks of supply, with excess units locking up cash.
3. **Watch / Volatile (Top-Right):** Erratic demand patterns requiring supply chain intervention.
4. **Healthy (Bottom-Left):** Balanced inventory matching consumer pull.

---

### Slide 5: Immediate Action Plan for Operations
1. **Raise Immediate Replenishment POs:**
   - Place purchase order for critical bestsellers (e.g. `NB-FUR-001`) with recommended reorder units to prevent complete stockouts over the next 21 days.
2. **Implement Clearance Tiers:**
   - Apply a 15–25% promotional bundle discount across top overstocked categories (e.g. slow-moving décor and kitchenware) to liquidate excess units and recover cash.
3. **Integrate FORESIGHT into Weekly S&OP:**
   - Replace monthly spreadsheets with the Streamlit Planning Dashboard for Monday inventory review meetings.

---

### Slide 6: Risk Governance & Model Monitoring
* **Quarterly Retraining:** Re-train the LightGBM models quarterly as new product categories launch.
* **Lead-Time Drift Tracking:** Periodically update supplier lead times in the inventory snapshot to account for logistics fluctuations.
* **Drift Alert:** If rolling 4-week WAPE degrades beyond 22%, trigger model inspection and hyperparameter re-tuning.

---

### Deliverable Acceptance & Handoff
All code, models, dashboards, and services are deployed and verifiable in the repository:
* **Planning Dashboard:** `streamlit run app/app.py`
* **Scoring REST Service:** `uvicorn service.main:app --port 8000`
* **Pipeline Re-runner:** `python run_pipeline.py`
