# Experimental Modeling Report: Resale Price Prediction

## Overview
This experiment aims to predict **resale_price** using a preprocessed training dataset containing both raw and engineered features.  
Unlike typical workflows, the provided test dataset lacks target labels (`resale_price`), so it was **excluded from model training and evaluation**.  
Instead, the training dataset was internally split (80% train / 20% validation) to assess model performance and ensure reliability.

Two models were selected for experimentation:

1. **Ridge Regression** — a regularized linear model for interpretable baseline analysis.  
2. **Random Forest Regressor** — a simple yet powerful tree-based ensemble for capturing nonlinear patterns.

These models represent two distinct learning paradigms — linear and nonlinear — providing complementary insights into feature behavior and predictive power.

---

## Data Description
The dataset includes both **original attributes** (e.g., flat model, year built, location) and **engineered features** such as:
- One-hot encoded categorical variables (e.g., `FLAT_TYPE_*`)
- Derived numerical features (e.g., `AGE = SELL_YEAR - BUILD_YEAR`)
- Geographical descriptors (`latitude`, `longitude`, and related location-based scores`)

Since some original columns (like `BLOCK`, `FLAT_MODEL`, or `TOWN`) are now represented by encoded or derived features, they were dropped to reduce redundancy.  
Similarly, features that are highly correlated (e.g., `BUILD_YEAR` and `SELL_YEAR` both influencing `AGE`) were down-weighted to avoid overemphasizing their contribution.

Missing values were filled with median values, and numerical features were standardized for linear model stability.

---

## Modeling Approach

### 1. Ridge Regression
Ridge Regression minimizes the sum of squared errors with an L2 penalty, which helps stabilize coefficients in the presence of multicollinearity.

**Advantages**
- Interpretable and mathematically simple  
- Works well with correlated predictors  
- Regularization reduces overfitting  

**Limitations**
- Captures only linear relationships  
- May underperform when the target depends on feature interactions  

---

### 2. Random Forest Regressor
Random Forest is an ensemble of decision trees built on random subsets of data and features, capable of capturing nonlinear and complex relationships.

**Advantages**
- Models nonlinear effects and feature interactions automatically  
- Robust to outliers and feature scaling  
- Provides direct feature importance estimation  

**Limitations**
- Less interpretable compared to linear models  
- Computationally heavier  
- Can overfit if not tuned (though mitigated here with depth limits)  

---

## Evaluation Strategy
Because no labeled test set was available, the **training data** was split into:
- **80% training subset**
- **20% validation subset**

The following evaluation metrics were calculated on the validation set:
- **R² (Coefficient of Determination):** Proportion of variance explained  
- **MAE (Mean Absolute Error):** Average magnitude of prediction error  
- **RMSE (Root Mean Squared Error):** Penalizes larger prediction errors  

All results are saved to:   

```
model_evaluation_results.csv
```

---

## Feature Importance
Two visualizations are generated to interpret the most influential predictors:

- `ridge_feature_importance.png`: Absolute coefficient magnitudes from Ridge Regression  
- `rf_feature_importance.png`: Feature importance scores from Random Forest  

These plots highlight the top 20 features most strongly correlated with **resale_price**, offering practical insights for future feature refinement.

---

## Notes and Future Improvements
This experimental setup intentionally focuses on **simplicity and interpretability**.  
However, several improvements could enhance model accuracy and generalization:

- Implement **cross-validation** for more stable metric estimation  
- Apply **hyperparameter optimization** (e.g., GridSearchCV or Optuna)  
- Explore **nonlinear gradient boosting models** such as **XGBoost**, **LightGBM**, or **CatBoost**  
- Conduct **residual analysis** to identify bias or heteroscedasticity  
- Consider **feature selection techniques** (e.g., mutual information or SHAP-based pruning)  

---

## Usage
Run the model training and evaluation with:

```bash
python model_train.py
```
