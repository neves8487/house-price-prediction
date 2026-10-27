import os
import pandas as pd
import numpy as np

def run_preprocessing(data_dir="data"):
    print("=" * 70)
    print("STARTING DATA PREPROCESSING & FEATURE ENGINEERING PIPELINE")
    print("=" * 70)

    # 1. Load raw datasets
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    print(f"Original records - Train: {train.shape[0]}, Test: {test.shape[0]}")

    # 2. Outlier Removal (recommended by dataset author for Ames Housing)
    # Homes with > 4,000 sq ft sold for unusually low prices (IDs 524 and 1299)
    outliers = train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index
    print(f"Removing {len(outliers)} extreme outliers from training set (GrLivArea > 4000 sq ft).")
    train = train.drop(outliers).reset_index(drop=True)

    # Store target and IDs
    train_ids = train['Id']
    test_ids = test['Id']
    y_train = train['SalePrice']
    y_train_log = np.log1p(y_train)  # log1p transformation to align with RMSLE metric

    # Concatenate features for 100% consistent preprocessing and encoding
    features_train = train.drop(['Id', 'SalePrice'], axis=1)
    features_test = test.drop(['Id'], axis=1)
    df = pd.concat([features_train, features_test], axis=0).reset_index(drop=True)
    n_train = len(features_train)

    # 3. Handle Missing Values
    # Categorical features where NA means "absence of feature" ("None")
    cols_none = [
        'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'MasVnrType'
    ]
    for col in cols_none:
        df[col] = df[col].fillna('None')

    # Numerical features where NA means zero (no garage, no basement, etc.)
    cols_zero = [
        'GarageArea', 'GarageCars', 'GarageYrBlt',
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
        'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'
    ]
    for col in cols_zero:
        df[col] = df[col].fillna(0)

    # LotFrontage: Impute by median of the corresponding neighborhood
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
        lambda val: val.fillna(val.median())
    )
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())

    # Sparse missing values in categorical columns -> impute with mode or standard default
    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
    df['MSZoning'] = df.groupby('MSSubClass')['MSZoning'].transform(
        lambda val: val.fillna(val.mode()[0] if not val.mode().empty else 'RL')
    )
    df['Utilities'] = df['Utilities'].fillna('AllPub')
    df['Functional'] = df['Functional'].fillna('Typ')
    df['KitchenQual'] = df['KitchenQual'].fillna('TA')
    df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])
    df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])
    df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])

    # 4. Convert numeric columns that are actually categorical
    df['MSSubClass'] = df['MSSubClass'].astype(str)
    df['MoSold'] = df['MoSold'].astype(str)

    # 5. Ordinal Quality Encoding
    qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
    qual_cols = [
        'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
        'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC'
    ]
    for col in qual_cols:
        df[col] = df[col].map(qual_map).fillna(0).astype(int)

    df['BsmtExposure'] = df['BsmtExposure'].map({'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None': 0}).fillna(0).astype(int)
    bsmt_fin_map = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'None': 0}
    df['BsmtFinType1'] = df['BsmtFinType1'].map(bsmt_fin_map).fillna(0).astype(int)
    df['BsmtFinType2'] = df['BsmtFinType2'].map(bsmt_fin_map).fillna(0).astype(int)
    df['GarageFinish'] = df['GarageFinish'].map({'Fin': 3, 'RFn': 2, 'Unf': 1, 'None': 0}).fillna(0).astype(int)
    df['Fence'] = df['Fence'].map({'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'None': 0}).fillna(0).astype(int)

    # 6. Domain-Driven Feature Engineering
    print("\nEngineering domain-specific real estate features...")

    # Total usable square footage (Basement + 1st Floor + 2nd Floor)
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

    # Total combined bathrooms (weighted by half baths)
    df['TotalBath'] = df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath'])

    # Property age metrics at sale time
    df['HouseAge'] = (df['YrSold'] - df['YearBuilt']).clip(lower=0)
    df['RemodAge'] = (df['YrSold'] - df['YearRemodAdd']).clip(lower=0)

    # Total porch and deck square footage
    df['TotalPorchSF'] = (
        df['WoodDeckSF'] + df['OpenPorchSF'] + 
        df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
    )

    # Interaction feature: overall quality score
    df['OverallScore'] = df['OverallQual'] * df['OverallCond']

    # Binary amenity indicators
    df['HasPool'] = (df['PoolArea'] > 0).astype(int)
    df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)
    df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
    df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)
    df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)

    # 7. One-Hot Encoding for remaining categorical variables
    df = pd.get_dummies(df, drop_first=True)
    df = df.astype(float)

    # 8. Split back into Train and Test sets
    train_clean = df.iloc[:n_train].copy()
    test_clean = df.iloc[n_train:].copy()

    # Add back IDs and target variable
    train_clean.insert(0, 'Id', train_ids.values)
    train_clean['SalePrice'] = y_train.values
    train_clean['SalePrice_log'] = y_train_log.values

    test_clean.insert(0, 'Id', test_ids.values)

    # 9. Save preprocessed datasets
    train_out = os.path.join(data_dir, "train_clean.csv")
    test_out = os.path.join(data_dir, "test_clean.csv")
    train_clean.to_csv(train_out, index=False)
    test_clean.to_csv(test_out, index=False)

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETED SUCCESSFULLY!")
    print(f"Final Train Shape: {train_clean.shape} -> Saved to {train_out}")
    print(f"Final Test Shape:  {test_clean.shape} -> Saved to {test_out}")
    print(f"Zero missing values in Train: {train_clean.isnull().sum().sum() == 0}")
    print(f"Zero missing values in Test:  {test_clean.isnull().sum().sum() == 0}")
    print("=" * 70)

    return train_clean, test_clean

if __name__ == "__main__":
    run_preprocessing()
