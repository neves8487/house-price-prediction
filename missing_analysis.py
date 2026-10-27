import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load train and test datasets
train_df = pd.read_csv(os.path.join("data", "train.csv"))
test_df = pd.read_csv(os.path.join("data", "test.csv"))

print("=" * 70)
print("MISSING VALUES ANALYSIS")
print("=" * 70)

# 2. Compute missing values in training set
total_rows = len(train_df)
nulls_train = train_df.isnull().sum()
nulls_train = nulls_train[nulls_train > 0].sort_values(ascending=False)

nulls_df = pd.DataFrame({
    'Missing_Count': nulls_train,
    'Percentage': (nulls_train / total_rows) * 100,
    'Type': [train_df[col].dtype for col in nulls_train.index]
})

print(f"\nFound {len(nulls_df)} columns with missing values in the training set (out of {train_df.shape[1]} total columns).\n")
print(f"{'Feature':<18} | {'Nulls':<7} | {'%':<6} | {'Type':<10} | {'Domain Meaning'}")
print("-" * 75)

# Mapping of NA meanings according to data_description.txt
meanings = {
    'PoolQC': 'No Pool ("None")',
    'MiscFeature': 'No special feature ("None")',
    'Alley': 'No alley access ("None")',
    'Fence': 'No Fence ("None")',
    'MasVnrType': 'No masonry veneer ("None")',
    'FireplaceQu': 'No Fireplace ("None")',
    'LotFrontage': 'Street frontage unmeasured (True missing)',
    'GarageType': 'No Garage ("None")',
    'GarageYrBlt': 'No Garage (Year missing)',
    'GarageFinish': 'No Garage ("None")',
    'GarageQual': 'No Garage ("None")',
    'GarageCond': 'No Garage ("None")',
    'BsmtExposure': 'No Basement ("None")',
    'BsmtFinType2': 'No Basement ("None")',
    'BsmtFinType1': 'No Basement ("None")',
    'BsmtCond': 'No Basement ("None")',
    'BsmtQual': 'No Basement ("None")',
    'MasVnrArea': 'Masonry veneer area (0 sq ft)',
    'Electrical': 'Missing electrical system (1 home in train)'
}

for col, row in nulls_df.iterrows():
    meaning = meanings.get(col, "Check data dictionary")
    print(f"{col:<18} | {int(row['Missing_Count']):<7} | {row['Percentage']:>5.1f}% | {str(row['Type']):<10} | {meaning}")

# 3. Save visual missing values chart
os.makedirs("plots", exist_ok=True)
output_img = os.path.join("plots", "analise_nulos.png")

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(12, 7))

sns.barplot(
    x=nulls_df['Percentage'], 
    y=nulls_df.index, 
    ax=ax, 
    hue=nulls_df.index,
    palette="flare_r",
    legend=False
)

ax.set_title("Percentage of Missing Values per Feature (Training Set)", fontsize=14, fontweight='bold')
ax.set_xlabel("% of Missing Rows", fontsize=11)
ax.set_ylabel("Feature Name", fontsize=11)

for i, v in enumerate(nulls_df['Percentage']):
    ax.text(v + 0.5, i, f"{v:.1f}%", va='center', fontsize=9, fontweight='bold', color='#333333')

plt.xlim(0, 105)
plt.tight_layout()
plt.savefig(output_img, dpi=150)
plt.close()

print("\n" + "=" * 70)
print(f"Chart successfully saved to: {output_img}")
print("=" * 70)
