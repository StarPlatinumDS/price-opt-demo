# 📘 Retail Price Optimizer – Manual & Usage Guide

Welcome! This is a simple guide to understand what this project does and how to use it.

---

## 🧠 What This Project Does

This project simulates a retail environment and helps determine the **optimal selling price** for a product based on predicted customer demand. It:

- Simulates realistic product sales data based on price
- Trains a machine learning model to predict demand (units sold)
- Calculates expected revenue at different price points
- Finds the price that maximizes revenue for each product and day
- Includes an interactive dashboard to experiment with pricing

---

## 🛠 How to Use It

### ▶️ 1. Run the Streamlit App (UI)
```bash
streamlit run streamlit_app.py
```
- Select a product (A–D)
- Choose a day of the week
- Optionally apply a discount
- View the revenue curve and optimal price

---

### 💻 2. Run the Logic from Python Script (CLI or notebook)
```python
from pricing_optimizer import (
    simulate_dataset,
    prepare_features,
    train_demand_model,
    simulate_price_curve,
    batch_optimize_prices
)

# Step 1: Simulate data
data = simulate_dataset()

# Step 2: Prepare features
features, target = prepare_features(data)

# Step 3: Train model
model, X_train, metrics = train_demand_model(features, target)

# Step 4: Simulate pricing curve for a single product/day
result = simulate_price_curve(model, X_train, product_id='B', day='Fri')

# Step 5: Batch optimize across all products/days
opt_df = batch_optimize_prices(model, X_train)
print(opt_df.head())
```

---

## 📂 Files Explained

| File/Folder         | Purpose                                     |
|---------------------|---------------------------------------------|
| `src/`              | All core Python logic                       |
| `streamlit_app.py`  | Interactive web app                         |
| `tests/`            | Unit tests with pytest                      |
| `requirements.txt`  | Python dependencies                        |
| `README.md`         | Project overview for GitHub                |
| `manual.md`         | You're reading it!                         |

---

## ✅ Quick Commands

| Action                 | Command                                  |
|------------------------|------------------------------------------|
| Install dependencies   | `pip install -r requirements.txt`        |
| Run app                | `streamlit run streamlit_app.py`         |
| Run tests              | `pytest tests/`                           |
| Build Docker image     | `docker build -t price-optimizer .`      |
| Run Docker container   | `docker run -p 8501:8501 price-optimizer`|

---

## 📦 Example Output

```
🏆 Optimal price: $62.22 — expected revenue: $1940.50
```