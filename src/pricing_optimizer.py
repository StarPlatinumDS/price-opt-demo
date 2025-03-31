import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def simulate_dataset(n_samples=1000, seed=42):
    np.random.seed(seed)
    data = pd.DataFrame({
        'product_id': np.random.choice(['A', 'B', 'C', 'D'], size=n_samples),
        'base_price': np.random.uniform(10, 100, size=n_samples),
        'discount': np.random.uniform(0, 0.3, size=n_samples),
        'day_of_week': np.random.choice(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], size=n_samples)
    })

    def price_effect(row):
        final_price = row['base_price'] * (1 - row['discount'])
        return max(0, 100 - final_price + np.random.normal(0, 5))

    data['units_sold'] = data.apply(price_effect, axis=1).astype(int)
    data['final_price'] = data['base_price'] * (1 - data['discount'])
    data['revenue'] = data['final_price'] * data['units_sold']
    return data

def prepare_features(data):
    data_fe = pd.get_dummies(data, columns=['product_id', 'day_of_week'], drop_first=True)
    features = data_fe.drop(columns=['units_sold', 'revenue'])
    target = data_fe['units_sold']
    return features, target

def train_demand_model(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    return model, X_train, metrics

def simulate_price_curve(model,
                         X_template,
                         product_id='A',
                         day='Fri',
                         discount=0.0,
                         price_range=np.linspace(10,100,90)
                         ):
    sim_data = pd.DataFrame({
        'base_price': price_range,
        'discount': discount
    })
    sim_data['final_price'] = sim_data['base_price'] * (1 - discount)

    for p in ['B', 'C', 'D']:
        sim_data[f'product_id_{p}'] = 1 if p == product_id else 0
    for d in ['Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']:
        sim_data[f'day_of_week_{d}'] = 1 if d == day else 0

    missing_cols = set(X_template.columns) - set(sim_data.columns)
    for col in missing_cols:
        sim_data[col] = 0

    sim_data = sim_data[X_template.columns]
    sim_data['predicted_units'] = np.clip(model.predict(sim_data), a_min=0, a_max=None)
    sim_data['revenue'] = sim_data['final_price'] * sim_data['predicted_units']

    return sim_data

def batch_optimize_prices(model, X_template):
    results = []
    product_ids = ['A', 'B', 'C', 'D']
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    for product in product_ids:
        for day in days:
            sim = simulate_price_curve(model, X_template, product_id=product, day=day)
            best_row = sim.loc[sim['revenue'].idxmax()]
            results.append({
                'product_id': product,
                'day_of_week': day,
                'optimal_price': best_row['final_price'],
                'expected_revenue': best_row['revenue'],
            })
    return pd.DataFrame(results).sort_values(['product_id', 'day_of_week'])

