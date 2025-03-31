# 🧠 Retail Price Optimization (Demo)

A data science project that simulates a retail pricing scenario and models price elasticity to recommend optimal prices for maximizing revenue. Built with a full MLOps-friendly structure and an interactive Streamlit app.

---

## 🚀 Project Features

- 📊 Simulates realistic retail pricing & sales data
- 🧠 Trains a demand prediction model (Linear Regression)
- 💸 Optimizes prices based on revenue simulations
- 📈 Batch optimizer for all product-day combos
- 🌐 Interactive Streamlit app to explore pricing strategies
- ✅ Pytest suite for unit testing core components

---

## 🗂️ Project Structure

```
price_optimization/
├── notebooks/                 # Original notebook
│   └── price_sim.ipynb
├── src/                        # Core logic (data sim, model, optimizer)
│   └── pricing_optimizer.py
├── tests/                     # Unit tests
│   └── test_pricing_optimizer.py
├── streamlit_app.py           # Interactive dashboard
├── requirements.txt           # Python dependencies
├── README.md                  # You're here!
├── manual.md                  # Guide on how to use the project
```

---

## 🛠️ Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/your-username/price_optimization.git
cd price_optimization
```

### 2. Create and activate environment
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run streamlit_app.py
```

### 4. Run tests
```bash
pytest tests/
```

---


## 🧪 Example Output
```text
🏆 Optimal price: $64.44 — expected revenue: $2189.75
```

---

