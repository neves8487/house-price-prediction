import os
import pandas as pd
import numpy as np

print("=" * 70)
print("INICIANDO PRÉ-PROCESSAMENTO & ENGENHARIA DE ATRIBUTOS")
print("=" * 70)

# 1. Carregar os ficheiros
train = pd.read_csv(os.path.join("data", "train.csv"))
test = pd.read_csv(os.path.join("data", "test.csv"))

print(f"Linhas originais - Treino: {train.shape[0]}, Teste: {test.shape[0]}")

# 2. Remoção de Outliers extremos (Recomendação documentada pelo criador do dataset)
# Casas com mais de 4000 pés quadrados vendidas a preços anormais (id 524 e 1299)
outliers = train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index
print(f"Removendo {len(outliers)} outliers extremos do treino (GrLivArea > 4000 com preço baixo).")
train = train.drop(outliers).reset_index(drop=True)

# Guardar target e IDs
train_ids = train['Id']
test_ids = test['Id']
y_train = train['SalePrice']
y_train_log = np.log1p(y_train) # Aplicar log(1 + x) para estabilizar a variância e alinhar com RMSLE

# Juntar dados de treino e teste para pré-processamento 100% consistente
features_train = train.drop(['Id', 'SalePrice'], axis=1)
features_test = test.drop(['Id'], axis=1)
df = pd.concat([features_train, features_test], axis=0).reset_index(drop=True)
n_train = len(features_train)

# 3. Tratamento de Valores em Falta (Imputação Baseada em Conhecimento do Negócio)

# A) Variáveis Categóricas onde NA significa "Ausência" ("None")
cols_none = [
    'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'MasVnrType'
]
for col in cols_none:
    df[col] = df[col].fillna('None')

# B) Variáveis Numéricas onde NA significa zero (sem garagem, sem cave, etc.)
cols_zero = [
    'GarageArea', 'GarageCars', 'GarageYrBlt',
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
    'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'
]
for col in cols_zero:
    df[col] = df[col].fillna(0)

# C) LotFrontage: Imputar pela mediana do bairro correspondente
df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
    lambda val: val.fillna(val.median())
)
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())

# D) Outras colunas com poucos nulos pontuais -> preencher com moda ou valor típico
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
df['MSZoning'] = df.groupby('MSSubClass')['MSZoning'].transform(
    lambda val: val.fillna(val.mode()[0] if not val.mode().empty else 'RL')
)
df['Utilities'] = df['Utilities'].fillna('AllPub')
df['Functional'] = df['Functional'].fillna('Typ') # data_description diz: assume typical
df['KitchenQual'] = df['KitchenQual'].fillna('TA')
df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])
df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])
df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])

# 4. Converter colunas numéricas que são na verdade categorias
# Ex: MSSubClass usa números (20, 60, 120) para indicar o tipo de habitação, não é uma quantidade!
df['MSSubClass'] = df['MSSubClass'].astype(str)
df['MoSold'] = df['MoSold'].astype(str)

# 5. Mapeamento de Variáveis Ordinais (Escala de Qualidade)
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

# 6. ENGENHARIA DE ATRIBUTOS (Criação de Variáveis Imobiliárias)
print("\nCriando novas variáveis com sentido de negócio...")

# Área total útil (cave + rés-do-chão + 2º andar)
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

# Total de casas de banho combinadas
df['TotalBath'] = df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath'])

# Idades no momento da transação (YrSold)
df['HouseAge'] = (df['YrSold'] - df['YearBuilt']).clip(lower=0)
df['RemodAge'] = (df['YrSold'] - df['YearRemodAdd']).clip(lower=0)

# Área exterior de alpendres e varandas
df['TotalPorchSF'] = (
    df['WoodDeckSF'] + df['OpenPorchSF'] + 
    df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
)

# Pontuação combinada de qualidade e condição
df['OverallScore'] = df['OverallQual'] * df['OverallCond']

# Indicadores binários (tem ou não tem)
df['HasPool'] = (df['PoolArea'] > 0).astype(int)
df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)
df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)
df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)

# 7. One-Hot Encoding para variáveis categóricas restantes
df = pd.get_dummies(df, drop_first=True)
# Garantir que todas as colunas booleanas são convertidas para int (0 ou 1)
df = df.astype(float)

# 8. Separar novamente Treino e Teste
train_clean = df.iloc[:n_train].copy()
test_clean = df.iloc[n_train:].copy()

# Adicionar IDs e target ao treino processado
train_clean.insert(0, 'Id', train_ids.values)
train_clean['SalePrice'] = y_train.values
train_clean['SalePrice_log'] = y_train_log.values

test_clean.insert(0, 'Id', test_ids.values)

# 9. Guardar datasets prontos para Machine Learning
train_out = os.path.join("data", "train_clean.csv")
test_out = os.path.join("data", "test_clean.csv")
train_clean.to_csv(train_out, index=False)
test_clean.to_csv(test_out, index=False)

print("\n" + "=" * 70)
print("PRÉ-PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
print(f"Shape Final do Treino: {train_clean.shape} -> Guardado em {train_out}")
print(f"Shape Final do Teste:  {test_clean.shape} -> Guardado em {test_out}")
print(f"Nenhum valor nulo restante no Treino: {train_clean.isnull().sum().sum() == 0}")
print(f"Nenhum valor nulo restante no Teste:  {test_clean.isnull().sum().sum() == 0}")
print("=" * 70)
