import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline

import lightgbm as lgb
import xgboost as xgb

print("=" * 70)
print("STARTING MODEL TRAINING & 5-FOLD CROSS-VALIDATION")
print("=" * 70)

# 1. Load preprocessed datasets
train = pd.read_csv(os.path.join("data", "train_clean.csv"))
test = pd.read_csv(os.path.join("data", "test_clean.csv"))

X = train.drop(['Id', 'SalePrice', 'SalePrice_log'], axis=1)
y = train['SalePrice_log']

X_test = test.drop(['Id'], axis=1)
test_ids = test['Id'].astype(int)

print(f"Features available for training: {X.shape[1]}")
print(f"Train samples: {X.shape[0]} | Test samples: {X_test.shape[0]}")

# 2. Configure 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 3. Define Baseline and Advanced Models
models = {
    "Ridge (Linear L2)": make_pipeline(RobustScaler(), Ridge(alpha=15.0, random_state=42)),
    "Lasso (Linear L1)": make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=42, max_iter=5000)),
    "LightGBM": lgb.LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    ),
    "XGBoost": xgb.XGBRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
}

# Containers for Out-Of-Fold (OOF) and Test predictions
oof_preds = {name: np.zeros(len(X)) for name in models}
test_preds = {name: np.zeros(len(X_test)) for name in models}
results = []

print("\n--- TRAINING INDIVIDUAL MODELS ---")

for name, model in models.items():
    print(f"\n>> Training {name}...")
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        # Fit model on current fold
        model.fit(X_tr, y_tr)
        
        # Predict on validation fold
        preds_val = model.predict(X_va)
        oof_preds[name][val_idx] = preds_val
        
        # Calculate fold RMSLE
        score = root_mean_squared_error(y_va, preds_val)
        fold_scores.append(score)
        
        # Accumulate test predictions across folds
        test_preds[name] += model.predict(X_test) / kf.n_splits
    
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    results.append({'Model': name, 'CV_RMSLE': mean_score, 'Std': std_score})
    print(f"   Mean RMSLE: {mean_score:.4f} (±{std_score:.4f})")

# 4. ENSEMBLE (Weighted Blend)
weights = {
    "Ridge (Linear L2)": 0.20,
    "Lasso (Linear L1)": 0.25,
    "LightGBM": 0.30,
    "XGBoost": 0.25
}

oof_blend = np.zeros(len(X))
test_blend = np.zeros(len(X_test))

for name, weight in weights.items():
    oof_blend += oof_preds[name] * weight
    test_blend += test_preds[name] * weight

blend_score = root_mean_squared_error(y, oof_blend)
results.append({'Model': '[ENSEMBLE] Weighted Blend', 'CV_RMSLE': blend_score, 'Std': 0.0})

# 5. FINAL COMPARATIVE TABLE
df_res = pd.DataFrame(results).sort_values(by='CV_RMSLE').reset_index(drop=True)

print("\n" + "=" * 70)
print("MODEL BENCHMARK RESULTS (RMSLE - Lower is Better)")
print("=" * 70)
print(f"{'Rank':<6} | {'Model':<30} | {'CV RMSLE':<10}")
print("-" * 55)
for i, row in df_res.iterrows():
    print(f" #{i+1:<4} | {row['Model']:<30} | {row['CV_RMSLE']:.4f}")
print("=" * 70)

# 6. GENERATE SUBMISSION FILE (submission.csv)
# Invert log1p transformation with expm1 to obtain actual dollar prices
final_prices = np.expm1(test_blend)

sub_df = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': final_prices
})

sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)

print(f"\nSUBMISSION FILE GENERATED SUCCESSFULLY: {sub_path}")
print(f" - Total rows: {len(sub_df)}")
print(f" - Min Price:    ${sub_df['SalePrice'].min():,.2f}")
print(f" - Mean Price:   ${sub_df['SalePrice'].mean():,.2f}")
print(f" - Max Price:    ${sub_df['SalePrice'].max():,.2f}")
print(f" - Missing values in submission: {sub_df.isnull().sum().sum()}")

# 7. FEATURE IMPORTANCE EXTRACTION
print("\nExtracting feature importance with LightGBM...")
lgb_final = lgb.LGBMRegressor(
    n_estimators=1000, learning_rate=0.03, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
)
lgb_final.fit(X, y)

importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': lgb_final.feature_importances_
}).sort_values(by='Importance', ascending=False).head(15)

os.makedirs("plots", exist_ok=True)
plot_imp_path = os.path.join("plots", "importancia_features.png")

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=importances, x='Importance', y='Feature', ax=ax, hue='Feature', palette='crest', legend=False)
ax.set_title("Top 15 Most Important Features (LightGBM)", fontsize=13, fontweight='bold')
ax.set_xlabel("Feature Importance (Split Count)")
ax.set_ylabel("Feature")
plt.tight_layout()
plt.savefig(plot_imp_path, dpi=150)
plt.close()

print(f"Feature importance plot saved to: {plot_imp_path}")
print("=" * 70)
