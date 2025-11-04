# ============================================================
# Experimental modeling for resale_price prediction
# Using Ridge Regression, Random Forest Regressor, and XGBoost
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

from xgboost import XGBRegressor

def main(data_dir: Path):
    print("===== [Step 1/7] Loading datasets... =====")
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
    print("===== [Step 2/7] Selecting features... =====")
    # Drop columns that are replaced or redundant
    drop_candidates = []

    drop_candidates += [col for col in train_df.columns if col.upper() in [
        "FLAT_TYPE_ORIGINAL", "FLAT_MODEL", "BLOCK", "TOWN", "STREET", "FLOOR_RANGE"
    ]]

    # Drop duplicates while preserving resale_price
    drop_candidates = list(set(drop_candidates) - {target_col})
    
    # Apply to train_df
    train_df = train_df.drop(columns=[c for c in drop_candidates if c in train_df.columns], errors="ignore")
    
    # Apply to test_df as well (to prepare for later predictions)
    test_features = test_df.drop(columns=[c for c in drop_candidates if c in test_df.columns], errors="ignore")

    # -------------------------------
    # Prepare data for modeling
    # -------------------------------  

    # 1. First, separate X and y
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]

    # 2. Execute train_test_split first
    print("===== [Step 3/7] Splitting data into train/validation... =====")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Handle missing values (using X_train's median only)
    print("===== [Step 4/7] Handling missing values... =====")
    train_median = X_train.median(numeric_only=True)
    X_train = X_train.fillna(train_median)
    X_val = X_val.fillna(train_median)
    X_test = test_features.fillna(train_median) # Fill the real test set

    # 4. Standardization
    print("===== [Step 5/7] Standardizing data... =====") 
    scaler = StandardScaler()
    scaler.fit(X_train) 
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    # X_test_scaled = scaler.transform(X_test) # (Needed if Ridge is the best model)

    print("===== [Step 6/7] Training models... =====")


    # ============================================================
    # Model 1: Ridge Regression
    # ============================================================
    print("\nTraining Ridge Regression...")
    ridge_model = Ridge(alpha=1.0, random_state=42)
    ridge_model.fit(X_train_scaled, y_train) # Use standardized data

    ridge_preds = ridge_model.predict(X_val_scaled) # Use standardized data
    ridge_r2 = r2_score(y_val, ridge_preds)
    ridge_mae = mean_absolute_error(y_val, ridge_preds)
    ridge_rmse = np.sqrt(mean_squared_error(y_val, ridge_preds))

    print("===== Ridge Regression Evaluation =====")
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
    print("Training Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        verbose=0 # Set to 1 to see progress
    )
    rf_model.fit(X_train, y_train) # Tree-based models don't require standardized data
    rf_preds = rf_model.predict(X_val)

    rf_r2 = r2_score(y_val, rf_preds)
    rf_mae = mean_absolute_error(y_val, rf_preds)
    rf_rmse = np.sqrt(mean_squared_error(y_val, rf_preds))

    print("===== Random Forest Evaluation =====")
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
    # Model 3: XGBoost Regressor
    # ============================================================
    print("Training XGBoost...")
    xgb_model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train, # Tree-based models don't require standardized data
                  eval_set=[(X_val, y_val)],
                  verbose=False) # Set to 100 to see progress

    xgb_preds = xgb_model.predict(X_val)
    xgb_r2 = r2_score(y_val, xgb_preds)
    xgb_mae = mean_absolute_error(y_val, xgb_preds)
    xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_preds))

    print("===== XGBoost Evaluation =====")
    print(f"R²:   {xgb_r2:.4f}")
    print(f"MAE:  {xgb_mae:.2f}")
    print(f"RMSE: {xgb_rmse:.2f}\n")

    xgb_importance = (
        pd.Series(xgb_model.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(10, 6))
    xgb_importance.head(20).plot(kind='barh', color='coral')
    plt.gca().invert_yaxis()
    plt.title("Top 20 Feature Importance – XGBoost (gain)")
    plt.xlabel("Importance (gain)")
    plt.tight_layout()
    plt.savefig(data_dir / "xgb_feature_importance.png")
    plt.close()


    print("===== [Step 7/7] Saving results... =====")
    # ============================================================
    # Save Results
    # ============================================================
    results = pd.DataFrame({
        "Model": ["Ridge Regression", "Random Forest", "XGBoost"],
        "R²": [ridge_r2, rf_r2, xgb_r2],
        "MAE": [ridge_mae, rf_mae, xgb_mae],
        "RMSE": [ridge_rmse, rf_rmse, xgb_rmse]
    })
    results_path = data_dir / "model_evaluation_results.csv"
    results.to_csv(results_path, index=False)
    print(f"Model comparison saved to {results_path}\n")
    print("===== Experiment Complete =====")


if __name__ == "__main__":
    output_dir = Path("data/test") 
    output_dir.mkdir(parents=True, exist_ok=True)
    main(output_dir)
