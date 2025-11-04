import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from scipy.stats import uniform, randint
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# Custom Scorers for Log-Transformed Target
# ============================================================

def r2_expm1(y_log_true, y_log_pred):
    """R² score on the original scale (after expm1)."""
    y_true = np.expm1(y_log_true)
    y_pred = np.expm1(y_log_pred)
    return r2_score(y_true, y_pred)

def mae_expm1(y_log_true, y_log_pred):
    """MAE on the original scale (after expm1)."""
    y_true = np.expm1(y_log_true)
    y_pred = np.expm1(y_log_pred)
    return mean_absolute_error(y_true, y_pred)

def rmse_expm1(y_log_true, y_log_pred):
    """RMSE on the original scale (after expm1)."""
    y_true = np.expm1(y_log_true)
    y_pred = np.expm1(y_log_pred)
    return np.sqrt(mean_squared_error(y_true, y_pred))


# ============================================================
# Data Loading Function
# ============================================================

def load_prepare_data(data_dir: Path):
    """Loads and prepares X, y for modeling."""
    print(" Loading data...")
    train_path = data_dir / "train_with_all_features.csv"
    train_df = pd.read_csv(train_path)
    target_col = "RESALE_PRICE"

    # Drop columns not used for modeling
    drop_candidates = [
        "FLAT_TYPE_ORIGINAL", "FLAT_MODEL", "BLOCK", "TOWN", "STREET", "FLOOR_RANGE"
    ]
    drop_candidates = [c for c in drop_candidates if c in train_df.columns]
    train_df = train_df.drop(columns=drop_candidates, errors="ignore")

    X = train_df.drop(columns=[target_col], errors="ignore")
    y = train_df[target_col]
    return X, y


# ============================================================
# Main Optimization, Evaluation, and Prediction Function
# ============================================================

def main():
    data_dir = Path("data/test")  

    # 1. Load data
    X, y = load_prepare_data(data_dir)

    # 2. Apply log transformation to the target variable
    y_log = np.log1p(y)
    print(f"Data loaded. X shape: {X.shape}, y_log shape: {y_log.shape}")

    # 3. Build the preprocessing and modeling Pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('xgb', XGBRegressor(
            random_state=42,
            n_jobs=-1,
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
        ))
    ])

    # 4. Define the hyperparameter search space
    param_dist = {
        'xgb__n_estimators': randint(400, 1000),
        'xgb__learning_rate': uniform(0.01, 0.09),
        'xgb__max_depth': randint(6, 12),
        'xgb__subsample': uniform(0.7, 0.3),
        'xgb__colsample_bytree': uniform(0.7, 0.3)
    }

    # 5. Define Cross-Validation and Scorers
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    scorers = {
        "r2_original": make_scorer(r2_expm1, greater_is_better=True),
        "mae_original": make_scorer(mae_expm1, greater_is_better=False),
        "rmse_original": make_scorer(rmse_expm1, greater_is_better=False),
    }

    # 6. Perform hyperparameter tuning using RandomizedSearchCV
    print("\n" + "="*70)
    print("Starting Hyperparameter Tuning (RandomizedSearchCV)...")
    print("="*70)

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=50,       # 50 iterations for tuning
        cv=5,            # Use 5-fold CV during tuning
        scoring=scorers["rmse_original"],  # Optimize based on original-scale RMSE
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    random_search.fit(X, y_log)

    # 7. Print best parameters and performance
    print("\n" + "="*70)
    print("Hyperparameter Tuning Complete")
    print("="*70)
    print("Best parameters found:")
    for k, v in random_search.best_params_.items():
        print(f"  {k}: {v}")
    
    # Handle negative score from 'make_scorer' if 'greater_is_better=False'
    best_rmse_score = -random_search.best_score_ if random_search.best_score_ < 0 else random_search.best_score_
    print(f"\nBest CV score (Original-scale RMSE): {best_rmse_score:.2f}")

    best_model_pipeline = random_search.best_estimator_

    # 8. Perform final 10-Fold evaluation with the best model
    print("\n" + "="*70)
    print("🔍 Final Model Evaluation (10-Fold CV on Original Price Scale)")
    print("="*70)

    final_kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    # Evaluate on original scale
    cv_r2 = cross_val_score(best_model_pipeline, X, y_log, cv=final_kfold, scoring=scorers["r2_original"], n_jobs=-1)
    cv_mae = cross_val_score(best_model_pipeline, X, y_log, cv=final_kfold, scoring=scorers["mae_original"], n_jobs=-1)
    cv_rmse = cross_val_score(best_model_pipeline, X, y_log, cv=final_kfold, scoring=scorers["rmse_original"], n_jobs=-1)

    mean_r2 = np.mean(cv_r2)
    mean_mae = -np.mean(cv_mae)   # Convert back to positive
    mean_rmse = -np.mean(cv_rmse) # Convert back to positive

    print(f"Mean R²:    {mean_r2:.4f} (Std: {np.std(cv_r2):.4f})")
    print(f"Mean MAE:   {mean_mae:.2f} (Std: {np.std(-cv_mae):.2f})")
    print(f"Mean RMSE:  {mean_rmse:.2f} (Std: {np.std(-cv_rmse):.2f})")
    print("="*70)


    # ============================================================
    # 9. Final Prediction and Submission
    # ============================================================
    print("\n" + "="*70)
    print("Generating Final Submission File...")
    print("="*70)

    # 9.1. Re-train the best model on the ENTIRE training dataset
    print("Step 9.1: Re-training best model on the ENTIRE training dataset...")
    best_model_pipeline.fit(X, y_log)
    print("Done.")

    # 9.2. Load and process the test data
    print("Step 9.2: Loading and processing test data...")
    test_path = data_dir / "test_with_all_features.csv"
    test_df = pd.read_csv(test_path)
    
    # Apply the exact same feature selection as in 'load_prepare_data'
    drop_candidates = [
        "FLAT_TYPE_ORIGINAL", "FLAT_MODEL", "BLOCK", "TOWN", "STREET", "FLOOR_RANGE"
    ]
    drop_candidates = [c for c in drop_candidates if c in test_df.columns]
    X_test_raw = test_df.drop(columns=drop_candidates, errors="ignore")
    
    # CRITICAL: Ensure X_test column order matches X_train (X)
    X_test = X_test_raw[X.columns]
    print(f"Test data prepared. Shape: {X_test.shape}")

    # 9.3. Generate predictions
    # The pipeline will automatically handle missing value imputation on X_test
    print("Step 9.3: Generating log-scale predictions...")
    log_predictions = best_model_pipeline.predict(X_test)
    
    # 9.4. Convert predictions back to the original price scale (expm1)
    print("Step 9.4: Converting predictions back to original price scale (expm1)...")
    final_predictions = np.expm1(log_predictions)

    # 9.5. Save the submission file
    submission = pd.DataFrame({
        "Id": test_df.index,
        "Predicted": final_predictions
    })
    submission_path = data_dir / "submission_xgboost_optimized.csv"
    submission.to_csv(submission_path, index=False)
    
    print("\n" + "="*70)
    print(f"Submission file saved to: {submission_path}")
    print("===== Experiment Complete =====")


if __name__ == "__main__":
    main()
