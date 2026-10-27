# 🏠 House Prices - Advanced Regression Techniques

An end-to-end Machine Learning pipeline solving the Kaggle [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) competition.

The objective is to predict residential sale prices (`SalePrice`) in Ames, Iowa, using 79 explanatory features covering physical, spatial, and structural characteristics of residential properties.

---

## 📌 Project Architecture

```bash
house_price/
├── data/
│   ├── train.csv               # Raw training dataset (1,460 homes)
│   ├── test.csv                # Raw test dataset (1,459 homes)
│   ├── data_description.txt    # Official Ames dataset dictionary
│   ├── train_clean.csv         # Processed training data (zero missing values)
│   └── test_clean.csv          # Processed test data (zero missing values)
├── plots/
│   ├── exploracao_inicial.png  # Exploratory Data Analysis & correlation plots
│   ├── analise_nulos.png       # Missing value ranking and diagnostics
│   └── importancia_features.png# Top 15 most important features (LightGBM)
├── eda.py                      # Exploratory Data Analysis (EDA) script
├── missing_analysis.py         # Missing data inspection & business logic mapping
├── preprocess.py               # Feature Engineering & data transformation pipeline
├── train.py                    # 5-Fold Cross-Validation, Ensemble Blending & inference
├── submission.csv              # Final formatted Kaggle submission file
├── requirements.txt            # Project dependencies
├── .gitignore
└── README.md
```

---

## 🔍 1. Exploratory Data Analysis (EDA)

Key statistical and exploratory findings:

- **Target (`SalePrice`)**: The target distribution is positively skewed (right-skewed), with values ranging between **$34,900** and **$755,000** (median: **$163,000**). To stabilize variance and optimize directly for the competition's **RMSLE** metric, a `np.log1p` transformation is applied.
- **Top Linear Correlates**:
  1. `OverallQual` (+0.791): Overall material and finish quality.
  2. `GrLivArea` (+0.709): Above-ground living area in sq ft.
  3. `GarageCars` (+0.640) / `GarageArea` (+0.623): Garage capacity and size.
  4. `TotalBsmtSF` (+0.614): Total basement area.
  5. `1stFlrSF` (+0.606): First-floor living area.
  6. `FullBath` (+0.561): Number of full bathrooms.
  7. `YearBuilt` (+0.523): Original construction year.

---

## 🛠️ 2. Data Cleaning & Feature Engineering

### A) Domain Imputation for Missing Values
- **Structural "Absence" NAs**: Categorical attributes such as `PoolQC`, `MiscFeature`, `Alley`, `Fence`, `FireplaceQu`, `GarageType`, and `BsmtQual` use `NA` to indicate the absence of the amenity (e.g., "No Pool", "No Garage"), rather than missing measurements. These were explicitly imputed as `'None'`.
- **Numerical NAs**: Missing numerical entries for garage and basement properties (`GarageArea`, `TotalBsmtSF`, etc.) were imputed as `0`.
- **LotFrontage**: Imputed using the median street frontage of the corresponding `Neighborhood`.

### B) Ordinal Encoding
- Qualitative features with explicit rating ranks (`Ex`, `Gd`, `TA`, `Fa`, `Po`, `None`) were converted to an integer scale from `5` down to `0`.
- `Fence` and basement finish quality types were mapped to appropriate ordinal integer scales.

### C) Domain-Specific Features
- `TotalSF`: Total usable property square footage ($\text{Basement} + \text{1stFlr} + \text{2ndFlr}$).
- `TotalBath`: Weighted total bathroom count ($\text{FullBath} + 0.5 \times \text{HalfBath} + \text{BsmtFullBath} + 0.5 \times \text{BsmtHalfBath}$).
- `HouseAge`: Property age at sale ($\text{YrSold} - \text{YearBuilt}$).
- `RemodAge`: Years since last renovation ($\text{YrSold} - \text{YearRemodAdd}$).
- `TotalPorchSF`: Combined square footage of all porch and deck areas.
- `OverallScore`: Interaction term multiplying overall quality by condition ($\text{OverallQual} \times \text{OverallCond}$).
- Binary amenity indicators: `HasPool`, `HasGarage`, `HasBasement`, `HasFireplace`, `Has2ndFloor`.

---

## 🏆 3. Modeling & 5-Fold Cross-Validation

Models were evaluated using a 5-Fold Cross-Validation strategy with the official **RMSLE** (*Root Mean Squared Log Error*) metric:

| Rank | Model | 5-Fold CV RMSLE | Notes |
| :---: | :--- | :---: | :--- |
| **🥇 #1** | **ENSEMBLE (Weighted Blend)** | **0.1115** | Optimal blend combining regularized linear & tree-based models |
| **🥈 #2** | **Lasso (Linear L1)** | **0.1148** | RobustScaler + L1 regularization for feature sparsity |
| **🥉 #3** | **Ridge (Linear L2)** | **0.1152** | RobustScaler + L2 regularization |
| **#4** | **XGBoost** | **0.1176** | Gradient Boosted Decision Trees (1,200 estimators) |
| **#5** | **LightGBM** | **0.1214** | Fast histogram-based Gradient Boosted Trees |

> **Key Takeaway**: The **Weighted Ensemble Blend** outperformed every individual model, demonstrating the variance-reduction benefits of ensembling linear regularized models with gradient boosted trees.

---

## 📊 4. Feature Importance

Feature importance extracted from tree splits confirms that **total usable area (`TotalSF`)**, **overall quality (`OverallQual`)**, **above ground living area (`GrLivArea`)**, and **property age (`HouseAge`)** are the strongest drivers in house price determination.

---

## 🚀 How to Run Locally

### 1. Clone the repository and set up environment:
```bash
git clone https://github.com/neves8487/house-price-prediction.git
cd house-price-prediction
python -m venv .venv
```

Activate the environment:
* **Windows (PowerShell):**
  ```powershell
  .\.venv\Scripts\activate
  ```
* **Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```

### 2. Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run the pipeline:
```bash
# 1. Exploratory data analysis (outputs plots/exploracao_inicial.png)
python eda.py

# 2. Missing values analysis (outputs plots/analise_nulos.png)
python missing_analysis.py

# 3. Data preprocessing & feature engineering (generates clean datasets)
python preprocess.py

# 4. Train models with 5-Fold CV, generate ensemble & submission.csv
python train.py
```

---

## 📤 Kaggle Submission
The generated `submission.csv` contains 1,459 test predictions formatted and ready for upload to the Kaggle competition evaluation engine.
