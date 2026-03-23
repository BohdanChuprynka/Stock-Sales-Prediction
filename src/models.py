import copy
import time

import lightgbm as lgb
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

from .config import LIGHTGBM_PARAMS, LIGHTGBM_NUM_ROUNDS, RANDOM_STATE
from .evaluate import evaluate_model


def get_default_models():
    """Return a dictionary of baseline models with descriptive names."""
    return {
        "Random Forest": RandomForestRegressor(n_estimators=30, max_depth=10, random_state=RANDOM_STATE),
        "Ridge": Ridge(),
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "LightGBM": "LightGBM",
    }


def run_lightgbm(X_train, y_train, X_val, y_val, params=None):
    """Train a LightGBM model and return predictions on validation data."""
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = params or LIGHTGBM_PARAMS
    bst = lgb.train(params, train_data, num_boost_round=LIGHTGBM_NUM_ROUNDS, valid_sets=[val_data])
    return bst.predict(X_val, num_iteration=bst.best_iteration)


def run_models(models, X_train, y_train, X_val, y_val):
    """Train and evaluate multiple models, returning a dict of metric results."""
    results = {}
    for model_name, model in models.items():
        if model_name == "LightGBM":
            y_pred = run_lightgbm(X_train, y_train, X_val, y_val)
            results[model_name] = evaluate_model(y_pred, y_val)
            continue

        print(f"Running {model_name}")
        start_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        results[model_name] = evaluate_model(y_pred, y_val)
        elapsed = time.time() - start_time
        print(f"{model_name} took {elapsed:.1f} seconds")

    return results


def ensemble_models(model, X_train, y_train, num_iters=5):
    """Train independent copies of a model multiple times for ensembling."""
    trained_models = []
    print("Running ensemble training...")
    for i in range(num_iters):
        model_copy = copy.deepcopy(model)
        start = time.time()
        model_copy.fit(X_train, y_train)
        trained_models.append(model_copy)
        print(f"  Model {i} took {time.time() - start:.1f} seconds")
    return trained_models


def predict_ensemble(models, X):
    """Average predictions from an ensemble of models."""
    predictions = [m.predict(X) for m in models]
    return np.mean(predictions, axis=0)
