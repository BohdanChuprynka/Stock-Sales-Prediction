"""
Store Sales Forecasting — Full Pipeline

Runs the entire workflow end-to-end:
  1. Download data from Kaggle
  2. Preprocess and engineer features
  3. Train and evaluate models
  4. Generate portfolio visualizations

Usage:
    python run_pipeline.py
"""

import os
import subprocess
import sys
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor

from src.config import (
    DATA_DIR,
    KAGGLE_DIR,
    IMAGES_DIR,
    TRAIN_PATH,
    VALIDATION_PATH,
    TEST_PATH,
    TRAIN_SPLIT_RATIO,
    RANDOM_STATE,
)
from src.preprocessing import (
    fill_missing_values,
    sync_data_column,
    get_categorical_features,
    add_wage_day,
    extract_date,
    one_hot_encoding,
    align_columns,
    method_preprocessing,
)
from src.evaluate import evaluate_model, feature_importance
from src.models import get_default_models, run_models, ensemble_models, predict_ensemble


# ---------------------------------------------------------------------------
# Step 1: Download data from Kaggle
# ---------------------------------------------------------------------------
def download_data():
    """Download and extract the Kaggle competition dataset.

    Downloads every time unless the extracted dataset already exists.
    """
    zip_name = "store-sales-time-series-forecasting.zip"
    expected_file = KAGGLE_DIR / "train.csv"

    if expected_file.exists():
        print(f"[1/5] Dataset already exists at {KAGGLE_DIR}, skipping download.")
        return

    print("[1/5] Downloading data from Kaggle...")
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", "store-sales-time-series-forecasting"],
        check=True,
    )

    print("      Extracting archive...")
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_name) as archive:
        archive.extractall(KAGGLE_DIR)
    os.remove(zip_name)
    print(f"      Data extracted to {KAGGLE_DIR}")


# ---------------------------------------------------------------------------
# Step 2: Preprocess data
# ---------------------------------------------------------------------------
def preprocess():
    """Load raw data, engineer features, and save preprocessed CSVs."""
    if TRAIN_PATH.exists():
        print(f"[2/5] Preprocessed data already exists at {DATA_DIR}, skipping.")
        return

    print("[2/5] Preprocessing data...")

    # Load raw data
    train_dataset = pd.read_csv(KAGGLE_DIR / "train.csv")
    test_data = pd.read_csv(KAGGLE_DIR / "test.csv")
    oil_data = pd.read_csv(KAGGLE_DIR / "oil.csv")
    holiday_events = pd.read_csv(KAGGLE_DIR / "holidays_events.csv")
    stores_csv = pd.read_csv(KAGGLE_DIR / "stores.csv")

    print(f"      Train shape: {train_dataset.shape}")
    print(f"      Test shape:  {test_data.shape}")

    # --- Oil data: sync dates and fill missing values ---
    oil_data = sync_data_column(oil_data, train_dataset, test_data)
    oil_data["oil_price"] = fill_missing_values(oil_data["dcoilwtico"])
    oil_data.drop(columns=["dcoilwtico"], inplace=True)

    # --- Convert dates ---
    for df in [train_dataset, test_data, oil_data, holiday_events]:
        df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")

    # --- Merge auxiliary datasets ---
    for aux, key in [(oil_data, "date"), (holiday_events, "date"), (stores_csv, "store_nbr")]:
        train_dataset = train_dataset.merge(aux, on=key, how="left")
        test_data = test_data.merge(aux, on=key, how="left")

    # --- Label-encode categorical features ---
    # Save the column list BEFORE encoding (label encoding makes them numeric,
    # so one-hot encoding needs the original names to find them later).
    categorical_features = get_categorical_features(train_dataset)
    encoder = LabelEncoder()
    train_dataset[categorical_features] = train_dataset[categorical_features].apply(encoder.fit_transform)
    test_data[categorical_features] = test_data[categorical_features].apply(encoder.fit_transform)

    print(f"      Missing values after merge — train: {train_dataset.isna().sum().sum()}, test: {test_data.isna().sum().sum()}")

    # --- Wage day feature ---
    train_dataset = add_wage_day(train_dataset)
    test_data = add_wage_day(test_data)

    # --- Drop id, extract date features, cast booleans ---
    train_dataset.drop(columns=["id"], inplace=True)
    test_data.drop(columns=["id"], inplace=True)

    extract_date(train_dataset, "date")
    extract_date(test_data, "date")

    train_dataset["transferred"] = train_dataset["transferred"].astype(int)
    train_dataset["WageDay"] = train_dataset["WageDay"].astype(int)
    test_data["transferred"] = test_data["transferred"].astype(int)
    test_data["WageDay"] = test_data["WageDay"].astype(int)

    # --- One-hot encode (pass the saved column list explicitly) ---
    train_dataset = one_hot_encoding(train_dataset, columns=categorical_features)
    test_data = one_hot_encoding(test_data, columns=categorical_features)

    # --- Encode any remaining non-numeric columns (e.g. 'description' from holidays) ---
    remaining_str = train_dataset.select_dtypes(exclude="number").columns.difference(["date"])
    if len(remaining_str) > 0:
        print(f"      Label-encoding remaining string columns: {list(remaining_str)}")
        for col in remaining_str:
            train_dataset[col] = train_dataset[col].fillna("missing").astype(str)
            test_data[col] = test_data[col].fillna("missing").astype(str)
            le = LabelEncoder()
            le.fit(pd.concat([train_dataset[col], test_data[col]]))
            train_dataset[col] = le.transform(train_dataset[col])
            test_data[col] = le.transform(test_data[col])

    # --- Align columns ---
    train_dataset, test_data = align_columns(train_dataset, test_data, fix_columns=True)
    print(f"      Final columns: {len(train_dataset.columns)}")

    # --- Train/validation split (temporal, 70/30) ---
    split = int(len(train_dataset) * TRAIN_SPLIT_RATIO)
    validation_dataset = train_dataset[split:]
    train_dataset = train_dataset[:split]

    # --- Undersample zero sales (method 3: keep 20%) ---
    train_dataset = method_preprocessing(train_dataset, method=3)

    print(f"      Train rows after undersampling: {len(train_dataset)}")
    print(f"      Validation rows: {len(validation_dataset)}")

    # --- Save ---
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_dataset.to_csv(TRAIN_PATH, index=False)
    validation_dataset.to_csv(VALIDATION_PATH, index=False)
    test_data.to_csv(TEST_PATH, index=False)
    print(f"      Saved preprocessed data to {DATA_DIR}")


# ---------------------------------------------------------------------------
# Step 3: Train and evaluate models
# ---------------------------------------------------------------------------
def train_and_evaluate():
    """Train baseline + ensemble models, return results DataFrame."""
    print("[3/5] Training and evaluating models...")

    train_df = pd.read_csv(TRAIN_PATH, low_memory=False)
    val_df = pd.read_csv(VALIDATION_PATH, low_memory=False)

    feature_names = [c for c in train_df.columns if c not in ("sales", "date")]

    X_train = train_df[feature_names].to_numpy()
    y_train = train_df["sales"].to_numpy()
    X_val = val_df[feature_names].to_numpy()
    y_val = val_df["sales"].to_numpy()

    print(f"      X_train: {X_train.shape}, X_val: {X_val.shape}")

    # --- Feature importance ---
    print("      Computing feature importance...")
    importance_df = feature_importance(X_train, y_train, feature_names, top_features=12)

    # --- Baseline models ---
    models = get_default_models()
    results = run_models(models, X_train, y_train, X_val, y_val)
    results = pd.DataFrame(results)

    # --- Ensemble ---
    print("      Training ensemble...")
    trained_ensemble = ensemble_models(
        DecisionTreeRegressor(random_state=RANDOM_STATE),
        X_train, y_train, num_iters=5,
    )
    y_pred = predict_ensemble(trained_ensemble, X_val)
    results["Ensemble (Decision Tree)"] = evaluate_model(y_pred, y_val)

    print("\n      === Results ===")
    print(results.T.to_string())
    print()

    return results, importance_df


# ---------------------------------------------------------------------------
# Step 4: Generate visualizations
# ---------------------------------------------------------------------------
def generate_visuals(results, importance_df):
    """Save portfolio-quality charts to images/."""
    print("[4/5] Generating visualizations...")
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # --- Feature importance ---
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["Features"], importance_df["Mutual_Gain"], color="#4C72B0")
    plt.xlabel("Mutual Information Gain")
    plt.ylabel("Feature")
    plt.title("Top Features by Mutual Information")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Saved {IMAGES_DIR / 'feature_importance.png'}")

    # --- Model comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    model_names = results.columns.tolist()
    mae_values = results.loc["MAE"].values.astype(float)
    r2_values = results.loc["R2"].values.astype(float)

    colors_mae = ["#DD5143" if v == min(mae_values) else "#4C72B0" for v in mae_values]
    colors_r2 = ["#DD5143" if v == max(r2_values) else "#4C72B0" for v in r2_values]

    axes[0].barh(model_names, mae_values, color=colors_mae)
    axes[0].set_xlabel("Mean Absolute Error (MAE)")
    axes[0].set_title("MAE by Model (lower is better)")
    axes[0].invert_yaxis()
    for i, v in enumerate(mae_values):
        axes[0].text(v + 5, i, f"{v:.1f}", va="center", fontsize=9)

    axes[1].barh(model_names, r2_values, color=colors_r2)
    axes[1].set_xlabel("R² Score")
    axes[1].set_title("R² by Model (higher is better)")
    axes[1].set_xlim(0, 1)
    axes[1].invert_yaxis()
    for i, v in enumerate(r2_values):
        axes[1].text(v + 0.01, i, f"{v:.4f}", va="center", fontsize=9)

    plt.suptitle("Model Comparison — Store Sales Forecasting", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.savefig(IMAGES_DIR / "thumbnail.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Saved {IMAGES_DIR / 'model_comparison.png'}")
    print(f"      Saved {IMAGES_DIR / 'thumbnail.png'}")

    # --- Actual vs Predicted (best model by R²) ---
    print("      Generating actual vs predicted plot...")
    train_df = pd.read_csv(TRAIN_PATH, low_memory=False)
    val_df = pd.read_csv(VALIDATION_PATH, low_memory=False)
    feature_cols = [c for c in train_df.columns if c not in ("sales", "date")]

    best_model_name = results.T["R2"].astype(float).idxmax()
    print(f"      Best model by R²: {best_model_name}")

    # Re-train best sklearn model for the scatter plot
    from sklearn.ensemble import RandomForestRegressor
    best_model = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=RANDOM_STATE)
    best_model.fit(train_df[feature_cols].to_numpy(), train_df["sales"].to_numpy())
    y_val = val_df["sales"].to_numpy()
    y_pred = best_model.predict(val_df[feature_cols].to_numpy())

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.hexbin(y_val, y_pred, gridsize=60, cmap="Blues", mincnt=1)
    max_val = max(y_val.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Sales")
    ax.set_ylabel("Predicted Sales")
    ax.set_title(f"Actual vs Predicted — {best_model_name}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Saved {IMAGES_DIR / 'actual_vs_predicted.png'}")


# ---------------------------------------------------------------------------
# Step 5: Summary
# ---------------------------------------------------------------------------
def print_summary(results):
    """Print a final summary of the pipeline run."""
    print("[5/5] Pipeline complete!\n")

    best_mae = results.T["MAE"].astype(float).idxmin()
    best_r2 = results.T["R2"].astype(float).idxmax()

    print(f"      Best MAE:  {best_mae} ({results[best_mae]['MAE']:.2f})")
    print(f"      Best R²:   {best_r2} ({results[best_r2]['R2']:.4f})")
    print()
    print("      Outputs:")
    print(f"        Preprocessed data  → {DATA_DIR}/")
    print(f"        Visualizations     → {IMAGES_DIR}/")
    print(f"          - model_comparison.png")
    print(f"          - feature_importance.png")
    print(f"          - actual_vs_predicted.png")
    print(f"          - thumbnail.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Store Sales Forecasting — Full Pipeline")
    print("=" * 60)
    print()

    download_data()
    preprocess()
    results, importance_df = train_and_evaluate()
    generate_visuals(results, importance_df)
    print_summary(results)


if __name__ == "__main__":
    main()
