# ============================================================
# CatBoost modeling for resale_price prediction
# 保留所有 string 特征，自动识别类别特征
# ============================================================
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

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
    # Separate target
    # -------------------------------
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]

    # -------------------------------
    # Identify numeric columns (excluding target)
    # -------------------------------
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # -------------------------------
    # Handle missing values
    # -------------------------------
    # 训练集：数值列填充中位数
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())

    # 测试集：只对“同时存在于 test 的数值列”进行同样填充
    X_test = test_df.copy()
    common_num_cols = [col for col in num_cols if col in X_test.columns]
    X_test[common_num_cols] = X_test[common_num_cols].fillna(X[num_cols].median())

    # -------------------------------
    # 自动识别类别特征 (object / string)
    # -------------------------------
    cat_features = X.select_dtypes(include=['object']).columns.tolist()
    print(f"Detected {len(cat_features)} categorical features: {cat_features}")

    # -------------------------------
    # Train/Val split
    # -------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ============================================================
    # CatBoost Model
    # ============================================================
    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        random_seed=42,
        loss_function='RMSE',
        verbose=100
    )

    cat_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        cat_features=cat_features,
        use_best_model=True
    )

    # ============================================================
    # Evaluation
    # ============================================================
    val_preds = cat_model.predict(X_val)
    r2 = r2_score(y_val, val_preds)
    mae = mean_absolute_error(y_val, val_preds)
    rmse = mean_squared_error(y_val, val_preds, squared=False)

    print("\n===== CatBoost Evaluation =====")
    print(f"R²:   {r2:.4f}")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}\n")

    # ============================================================
    # Feature importance
    # ============================================================
    feature_importance = cat_model.get_feature_importance()
    fi_df = pd.DataFrame({
        'feature': X.columns,
        'importance': feature_importance
    }).sort_values(by='importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(fi_df['feature'].head(20), fi_df['importance'].head(20))
    plt.gca().invert_yaxis()
    plt.title("Top 20 Feature Importance - CatBoost")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(data_dir / "catboost_feature_importance.png")
    plt.close()


    # ============================================================
    # Test predictions
    # ============================================================
    final_predictions = cat_model.predict(X_test)

    submission = pd.DataFrame({
        'Id': X_test.index,
        'Predicted': final_predictions
    })
    submission.to_csv(data_dir / "submission_catboost.csv", index=False)

    print("✅ CatBoost training & validation complete.")
    print("✅ Saved model_evaluation_results_catboost.csv")
    print("✅ Saved submission_catboost.csv")
    print("✅ Saved catboost_feature_importance.png")

if __name__ == "__main__":
    output_dir = Path("../data/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    main(output_dir)
