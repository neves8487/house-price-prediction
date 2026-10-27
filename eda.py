import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load training data
data_path = os.path.join("data", "train.csv")
df = pd.read_csv(data_path)

print("=" * 60)
print("DATA LOADED SUCCESSFULLY!")
print(f"Total houses in training set: {df.shape[0]}")
print(f"Total features (columns): {df.shape[1]}")
print("=" * 60)

# 2. Target summary: SalePrice
target = df['SalePrice']
print("\n--- 1. SALE PRICE SUMMARY (SalePrice) ---")
print(f"Min Price:    ${target.min():,}")
print(f"Median Price: ${target.median():,}")
print(f"Mean Price:   ${target.mean():,.2f}")
print(f"Max Price:    ${target.max():,}")
print(f"Std Dev:      ${target.std():,.2f}")

# 3. Compute Pearson correlation with SalePrice
numeric_df = df.select_dtypes(include=[np.number])
correlations = numeric_df.corr()['SalePrice'].sort_values(ascending=False)

print("\n--- 2. TOP 10 FEATURES MOST CORRELATED WITH SALE PRICE ---")
print("Note: 1.0 is perfect positive correlation; > 0.60 indicates a strong relationship.")
top_10 = correlations.iloc[1:11]  # Exclude SalePrice itself (1.0)
for rank, (col, corr_val) in enumerate(top_10.items(), 1):
    print(f" {rank:2d}. {col:<15} -> Correlation: {corr_val:+.3f}")

# 4. Generate visual plots
os.makedirs("plots", exist_ok=True)
output_img = os.path.join("plots", "exploracao_inicial.png")

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle("Exploratory Data Analysis: Key Drivers of House Prices", fontsize=16, fontweight='bold')

# Plot 1: Target Distribution
sns.histplot(df['SalePrice'], kde=True, ax=axes[0, 0], color='#2b5c8f')
axes[0, 0].set_title("1. SalePrice Distribution (Positive / Right-Skewed)", fontsize=12)
axes[0, 0].set_xlabel("Sale Price ($)")
axes[0, 0].set_ylabel("Frequency")

# Plot 2: Top 10 Correlations
sns.barplot(x=top_10.values, y=top_10.index, ax=axes[0, 1], hue=top_10.index, palette="crest", legend=False)
axes[0, 1].set_title("2. Top 10 Features with Strongest Linear Correlation", fontsize=12)
axes[0, 1].set_xlabel("Pearson Correlation Coefficient")

# Plot 3: Living Area vs SalePrice
sns.scatterplot(data=df, x='GrLivArea', y='SalePrice', ax=axes[1, 0], alpha=0.6, color='#e76f51')
axes[1, 0].set_title("3. Above Ground Living Area (GrLivArea) vs Sale Price", fontsize=12)
axes[1, 0].set_xlabel("Living Area (sq ft)")
axes[1, 0].set_ylabel("Sale Price ($)")

# Plot 4: Overall Quality vs SalePrice
sns.boxplot(data=df, x='OverallQual', y='SalePrice', ax=axes[1, 1], hue='OverallQual', palette="Blues", legend=False)
axes[1, 1].set_title("4. Overall Material & Finish Quality (OverallQual: 1 to 10) vs Sale Price", fontsize=12)
axes[1, 1].set_xlabel("Overall Quality Rating (1 to 10)")
axes[1, 1].set_ylabel("Sale Price ($)")

plt.tight_layout()
plt.savefig(output_img, dpi=150)
plt.close()

print("\n" + "=" * 60)
print(f"Plot generated and saved successfully to: {output_img}")
print("=" * 60)
