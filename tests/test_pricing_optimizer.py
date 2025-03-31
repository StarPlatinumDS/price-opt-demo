import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pytest
import numpy as np
import pandas as pd
from pricing_optimizer import (
    simulate_dataset,
    prepare_features,
    train_demand_model,
    simulate_price_curve,
    batch_optimize_prices
)

def test_simulate_dataset():
    data = simulate_dataset(n_samples=100)
    assert isinstance(data, pd.DataFrame)
    assert all(col in data.columns for col in ['base_price', 'discount', 'units_sold', 'revenue'])
    assert len(data) == 100

def test_prepare_features():
    data = simulate_dataset(n_samples=100)
    features, target = prepare_features(data)
    assert features.shape[0] == target.shape[0] == 100
    assert 'final_price' in features.columns

def test_train_demand_model():
    data = simulate_dataset(n_samples=200)
    features, target = prepare_features(data)
    model, X_train, metrics = train_demand_model(features, target)
    assert 'mse' in metrics and 'r2' in metrics
    assert X_train.shape[1] == features.shape[1]

def test_simulate_price_curve():
    data = simulate_dataset(n_samples=300)
    features, target = prepare_features(data)
    model, X_template, _ = train_demand_model(features, target)
    sim_df = simulate_price_curve(model, X_template, product_id='C', day='Wed')
    assert 'revenue' in sim_df.columns
    assert sim_df['predicted_units'].min() >= 0

def test_batch_optimize_prices():
    data = simulate_dataset(n_samples=500)
    features, target = prepare_features(data)
    model, X_template, _ = train_demand_model(features, target)
    opt_df = batch_optimize_prices(model, X_template)
    assert isinstance(opt_df, pd.DataFrame)
    assert set(['product_id', 'day_of_week', 'optimal_price', 'expected_revenue']).issubset(opt_df.columns)
    assert opt_df.shape[0] == 28  # 4 products * 7 days
