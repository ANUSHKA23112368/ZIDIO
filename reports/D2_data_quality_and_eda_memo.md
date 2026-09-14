# MEMORANDUM

**TO:** Head of Operations, NorthBay Living  
**FROM:** Lead Data Scientist, Zidio Development  
**DATE:** September 14, 2026  
**SUBJECT:** Deliverable D2: Data Quality Audit & Exploratory Demand Analysis (Project FORESIGHT)  

---

### 1. Executive Summary
NorthBay Living operates an omnichannel catalog of 200 active SKUs across Furnishings, Décor, Small Appliances, and Kitchenware. Prior to this engagement, inventory planning relied on spreadsheet heuristics, leading to concurrent stockouts on high-velocity items and cash lockups on slow movers. 

This audit reviewed **150,801 daily sales transactions** across 821 calendar days (2024 to 2026), unified against SKU master dimensions and inventory snapshots. The automated pipeline successfully remediated data inconsistencies, resolved negative return anomalies, normalized disparate naming conventions, and constructed a robust, weekly demand foundation.

---

### 2. Data Quality Findings & Remediation Actions

| Issue Identified | Impact Before Fix | Remediation Strategy Applied | Status |
| :--- | :--- | :--- | :--- |
| **Transaction Duplicates** | 450 duplicate rows artificially inflated transaction counts | Automated deduplication applied during stage-1 ingestion | **Resolved** |
| **Product Return Records** | 40 instances of negative units sold (`units_sold < 0`) caused negative revenue figures | Returns aggregated into net weekly units, bounded at zero demand with separate return tracking | **Resolved** |
| **Category Casing Inconsistencies** | Mixed casing (`furnishings`, `FURNISHINGS`, ` Furnishings `) fragmented reporting | Standardized with `.str.strip().str.title()` across catalog | **Resolved** |
| **Missing Promo Indicators** | ~750 transaction rows lacked promotional flags | Inferred by cross-referencing against verified promotional calendar events | **Resolved** |
| **Missing Unit Costs** | 3 SKU master records omitted manufacturing/wholesale costs | Imputed using category median margin ratios | **Resolved** |

---

### 3. Demand Patterns & Business Insights

#### Insight 1: Heavy Pareto Concentration (The 80/20 Rule)
* **Finding:** The top 15% of SKUs generate **68.4% of total retail turnover**, while the bottom 30% of SKUs contribute under 4.8% of revenue.
* **Business Implication:** Stockouts on the top 30 hero items directly imperil NorthBay's revenue stream. Conversely, holding deep inventory buffers across long-tail décor items creates massive working capital drag.
* **Action:** Tier inventory planning by SKU velocity: maintain 98% service levels on high-velocity items with weekly PO reviews, and switch bottom-tier items to low-holding, make-to-order, or drop-ship schedules.

#### Insight 2: Festive Demand Surge & Price Elasticity Lift
* **Finding:** Sales demand surges by **160% to 220%** during October/November (Diwali Festive Window) and late November (Cyber Week), with a secondary weekend lift of 35%. Promotional events generate an average unit lift of 2.2x when discounted by 10–25%.
* **Business Implication:** Relying on rolling 4-week averages during August/September guarantees severe stockouts by Diwali week because past autumn demand is blind to seasonal peaks.
* **Action:** Incorporate calendar event and holiday markers directly into the demand forecasting model to trigger proactive procurement 6 to 8 weeks ahead of festival peaks.

#### Insight 3: Disproportionate Capital Trapped in Dead & Slow-Moving Stock
* **Finding:** Analysis of warehouse snapshots revealed 172 SKUs with **greater than 14 weeks of supply**, and many holding over 30 to 50 weeks of supply.
* **Business Implication:** NorthBay currently holds **₹37.49 Crores (₹374,964,818)** in inventory exceeding an 8-week operating cushion. This capital is physically degrading in the single warehouse and incurring storage overhead.
* **Action:** Establish immediate clearance promotional tiers (e.g., 20% bundle discount, seasonal flash sales) for items flagged in the **"Markdown / Clear"** quadrant to unlock liquid capital before peak purchasing cycles.

---

### 4. Acceptance Criteria Checklist (Deliverable D2)
- [x] **Data Quality Issues Reported:** Full audit log exported to `reports/pipeline_audit.json`.
- [x] **Demand Patterns Analyzed:** Seasonality, trend, Pareto velocity, and dead stock characterized.
- [x] **Plain Language Insights:** 3 actionable, non-technical business recommendations delivered.
- [x] **Reproducibility:** Pipeline regenerates all cleaned views end-to-end via `src/pipeline.py`.
