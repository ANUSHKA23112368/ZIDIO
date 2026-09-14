import sys
import io
import requests
import json

# Ensure UTF-8 output on Windows terminal
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=== 1. Health Check Endpoint ===")
r = requests.get("http://127.0.0.1:8000/health")
print("Status:", r.status_code)
print("Payload:", json.dumps(r.json(), indent=2))

print("\n=== 2. Real-Time Scoring for SKU: NB-FUR-001 ===")
r = requests.get("http://127.0.0.1:8000/score/NB-FUR-001")
data = r.json()
print(f"Product: {data['sku_id']} - {data['category']} > {data['subcategory']}")
print(f"Unit Cost: INR {data['unit_cost']} | List Price: INR {data['list_price']}")
print(f"Stock Position: On Hand = {data['on_hand_units']} | On Order = {data['on_order_units']} | Lead Time = {data['lead_time_days']} days")
print(f"Decision Quadrant: {data['quadrant']} ({data['quadrant_meaning']})")
print(f"Recommended Action: {data['recommended_action']}")
print(f"Sales at Risk: INR {data['sales_at_risk_inr']:,.2f}")
print(f"Suggested Reorder PO: {data['recommended_reorder_units']} units")
print("First 3 Weeks Forecast:")
for f in data["weekly_forecast"][:3]:
    print(f"  Week {f['week_horizon']} ({f['forecast_week']}): {f['predicted_units']} units (80% CI: [{f['lower_ci_80']} - {f['upper_ci_80']}])")

print("\n=== 3. What-If Simulation Endpoint ===")
sim_payload = {
    "sku_id": "NB-FUR-001",
    "custom_on_hand": 8,
    "custom_on_order": 0,
    "custom_lead_time_days": 30,
    "target_service_level": 0.98
}
r = requests.post("http://127.0.0.1:8000/simulate", json=sim_payload)
print("Simulation Result (Custom On-Hand=8, Lead Time=30 days):")
print(json.dumps(r.json(), indent=2))

print("\n=== 4. Testing Error Handling (Graceful 404 for Unknown SKU) ===")
r = requests.get("http://127.0.0.1:8000/score/NON_EXISTENT_SKU")
print("Status Code:", r.status_code)
print("Error Message:", r.json())
