# ============================================================
# Experimental modeling for resale_price prediction
# Using Ridge Regression and Random Forest Regressor
# ============================================================
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def main(data_dir: Path):
    # -------------------------------
    # Load dataset
    # -------------------------------
    train_path = data_dir / "train_with_all_features.csv"
    test_path = data_dir / "test_with_all_features.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)


    target_col = "RESALE_PRICE"

    # -------------------------------
    # Feature Selection Logic
    # -------------------------------
    # Drop columns that are replaced or redundant
    drop_candidates = []

    drop_candidates += [col for col in train_df.columns if col.upper() in [
        "FLAT_TYPE_ORIGINAL", "FLAT_MODEL", "BLOCK", "TOWN", "STREET", "MONTH_NUM", "FLOOR_RANGE"
    ]]


    # Drop duplicates while preserving resale_price
    drop_candidates = list(set(drop_candidates) - {target_col})
    train_df = train_df.drop(columns=[c for c in drop_candidates if c in train_df.columns], errors="ignore")
    test_df = test_df.drop(columns=[c for c in drop_candidates if c in test_df.columns], errors="ignore")

    # -------------------------------
    # Handle missing values
    # -------------------------------
    train_df = train_df.fillna(train_df.median(numeric_only=True))
    test_df = test_df.fillna(train_df.median(numeric_only=True))

    # -------------------------------
    # Separate features and target
    # -------------------------------
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------------
    # Standardize numeric features
    # -------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # ============================================================
    # Model 1: Ridge Regression
    # ============================================================
    ridge_model = Ridge(alpha=1.0, random_state=42)
    ridge_model.fit(X_train_scaled, y_train)

    ridge_preds = ridge_model.predict(X_val_scaled)
    ridge_r2 = r2_score(y_val, ridge_preds)
    ridge_mae = mean_absolute_error(y_val, ridge_preds)
    ridge_rmse = np.sqrt(mean_squared_error(y_val, ridge_preds))

    print("===== Ridge Regression Evaluation (Train only) =====")
    print(f"R²:   {ridge_r2:.4f}")
    print(f"MAE:  {ridge_mae:.2f}")
    print(f"RMSE: {ridge_rmse:.2f}\n")

    # Feature importance
    ridge_importance = pd.Series(np.abs(ridge_model.coef_), index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    ridge_importance.head(20).plot(kind='barh', color='skyblue')
    plt.gca().invert_yaxis()
    plt.title("Top 20 Feature Importance - Ridge Regression")
    plt.xlabel("Coefficient Magnitude")
    plt.tight_layout()
    plt.savefig(data_dir / "ridge_feature_importance.png")
    plt.close()

    # ============================================================
    # Model 2: Random Forest Regressor
    # ============================================================
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_val)

    rf_r2 = r2_score(y_val, rf_preds)
    rf_mae = mean_absolute_error(y_val, rf_preds)
    rf_rmse = np.sqrt(mean_squared_error(y_val, rf_preds))

    print("===== Random Forest Evaluation (Train only) =====")
    print(f"R²:   {rf_r2:.4f}")
    print(f"MAE:  {rf_mae:.2f}")
    print(f"RMSE: {rf_rmse:.2f}\n")

    rf_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    rf_importance.head(20).plot(kind='barh', color='seagreen')
    plt.gca().invert_yaxis()
    plt.title("Top 20 Feature Importance - Random Forest")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(data_dir / "rf_feature_importance.png")
    plt.close()

    # ============================================================
    # Save Results
    # ============================================================
    results = pd.DataFrame({
        "Model": ["Ridge Regression", "Random Forest"],
        "R²": [ridge_r2, rf_r2],
        "MAE": [ridge_mae, rf_mae],
        "RMSE": [ridge_rmse, rf_rmse]
    })
    results.to_csv(data_dir / "model_evaluation_results.csv", index=False)

    print("Model training and validation complete.")
    print("Results saved to 'model_evaluation_results.csv'.")
