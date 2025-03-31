import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from src.pricing_optimizer import simulate_dataset, prepare_features, train_demand_model, simulate_price_curve

#title
st.title("🧠 Retail Price Optimization App")
st.markdown("Optimize product pricing based on predicted demand and maximize revenue.")

#simulate & train model on app load
data = simulate_dataset()
features, target = prepare_features(data)
model, X_template, metrics = train_demand_model(features, target)

#sidebar
st.sidebar.header("🛠️ Simulation Settings")
product = st.sidebar.selectbox("Select Product", ['A', 'B', 'C', 'D'])
day = st.sidebar.selectbox("Select Day of Week", ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
discount = st.sidebar.slider("Discount (%)", 0, 30, 0) / 100

price_min, price_max = st.sidebar.slider("Price Range", 10, 100, (10, 100))

#simulate price curve
price_range = np.linspace(price_min, price_max, 100)
sim_df = simulate_price_curve(model, X_template, product_id=product, day=day, discount=discount, price_range=price_range)

#find opt price
best_row = sim_df.loc[sim_df['revenue'].idxmax()]

#Results
st.subheader(f"📊 Revenue Curve for Product {product} on {day}")
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(sim_df['final_price'], sim_df['revenue'])
ax.set_xlabel("Price ($)")
ax.set_ylabel("Predicted Revenue ($)")
ax.set_title("Revenue vs. Price")
ax.grid(True)
st.pyplot(fig)

st.success(f"🏆 Optimal Price: ${best_row['final_price']:.2f} — Expected Revenue: ${best_row['revenue']:.2f}")

#show table
with st.expander("🔍 View Full Simulation Data"):
    st.dataframe(sim_df[['final_price', 'predicted_units', 'revenue']].round(2))