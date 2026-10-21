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
print("INICIANDO TREINO DE MODELOS E VALIDAÇÃO CRUZADA (5-FOLD CV)")
print("=" * 70)

# 1. Carregar os dados limpos
train = pd.read_csv(os.path.join("data", "train_clean.csv"))
test = pd.read_csv(os.path.join("data", "test_clean.csv"))

X = train.drop(['Id', 'SalePrice', 'SalePrice_log'], axis=1)
y = train['SalePrice_log']

X_test = test.drop(['Id'], axis=1)
test_ids = test['Id'].astype(int)

print(f"Features disponíveis para treino: {X.shape[1]}")
print(f"Amostras de treino: {X.shape[0]} | Amostras de teste: {X_test.shape[0]}")

# 2. Configurar 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 3. Definir os Modelos
modelos = {
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

# Dicionários para guardar previsões OOF (Out-of-Fold) e previsões de teste
oof_preds = {nome: np.zeros(len(X)) for nome in modelos}
test_preds = {nome: np.zeros(len(X_test)) for nome in modelos}
resultados = []

print("\n--- TREINANDO MODELOS INDIVIDUAIS ---")

for nome, modelo in modelos.items():
    print(f"\n>> Treinando {nome}...")
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        # Treinar modelo no fold
        modelo.fit(X_tr, y_tr)
        
        # Previsão no fold de validação
        preds_val = modelo.predict(X_va)
        oof_preds[nome][val_idx] = preds_val
        
        # Calcular RMSLE do fold
        score = root_mean_squared_error(y_va, preds_val)
        fold_scores.append(score)
        
        # Acumular previsões no teste (média dos 5 folds)
        test_preds[nome] += modelo.predict(X_test) / kf.n_splits
    
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    resultados.append({'Modelo': nome, 'RMSLE Médio': mean_score, 'Desvio Padrão': std_score})
    print(f"   Score Médio: {mean_score:.4f} (±{std_score:.4f})")

# 4. CRIAR O ENSEMBLE (Média Ponderada dos Melhores)
# Pesos equilibrados entre modelos lineares e de árvores
pesos = {
    "Ridge (Linear L2)": 0.20,
    "Lasso (Linear L1)": 0.25,
    "LightGBM": 0.30,
    "XGBoost": 0.25
}

oof_blend = np.zeros(len(X))
test_blend = np.zeros(len(X_test))

for nome, peso in pesos.items():
    oof_blend += oof_preds[nome] * peso
    test_blend += test_preds[nome] * peso

blend_score = root_mean_squared_error(y, oof_blend)
resultados.append({'Modelo': '[ENSEMBLE] Blend Ponderado', 'RMSLE Médio': blend_score, 'Desvio Padrão': 0.0})

# 5. TABELA COMPARATIVA FINAL
df_res = pd.DataFrame(resultados).sort_values(by='RMSLE Médio').reset_index(drop=True)

print("\n" + "=" * 70)
print("TABELA COMPARATIVA DE RESULTADOS (RMSLE - Menor é Melhor)")
print("=" * 70)
print(f"{'Posição':<8} | {'Modelo':<30} | {'RMSLE CV':<10}")
print("-" * 55)
for i, row in df_res.iterrows():
    print(f" #{i+1:<6} | {row['Modelo']:<30} | {row['RMSLE Médio']:.4f}")
print("=" * 70)

# 6. GERAR O FICHEIRO DE SUBMISSÃO (submission.csv)
# Reverter a transformação logarítmica com np.expm1 para obter o preço em dólares
final_prices = np.expm1(test_blend)

sub_df = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': final_prices
})

sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)

print(f"\nFICHEIRO DE SUBMISSÃO GERADO COM SUCESSO: {sub_path}")
print(f" - Total de linhas: {len(sub_df)}")
print(f" - Preço Mínimo Previsto:  ${sub_df['SalePrice'].min():,.2f}")
print(f" - Preço Médio Previsto:   ${sub_df['SalePrice'].mean():,.2f}")
print(f" - Preço Máximo Previsto:  ${sub_df['SalePrice'].max():,.2f}")
print(f" - Valores nulos na submissão: {sub_df.isnull().sum().sum()}")

# 7. IMPORTÂNCIA DAS CARACTERÍSTICAS (FEATURE IMPORTANCE)
# Treinar o LightGBM nos dados completos para extrair a importância das features
print("\nExtraindo a importância das características com o LightGBM...")
lgb_final = lgb.LGBMRegressor(
    n_estimators=1000, learning_rate=0.03, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
)
lgb_final.fit(X, y)

importancias = pd.DataFrame({
    'Feature': X.columns,
    'Importancia': lgb_final.feature_importances_
}).sort_values(by='Importancia', ascending=False).head(15)

os.makedirs("plots", exist_ok=True)
plot_imp_path = os.path.join("plots", "importancia_features.png")

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=importancias, x='Importancia', y='Feature', ax=ax, hue='Feature', palette='crest', legend=False)
ax.set_title("Top 15 Características Mais Decisivas na Previsão (LightGBM)", fontsize=13, fontweight='bold')
ax.set_xlabel("Importância do Atributo (Número de divisões nas árvores)")
ax.set_ylabel("Característica")
plt.tight_layout()
plt.savefig(plot_imp_path, dpi=150)
plt.close()

print(f"Gráfico de importância guardado em: {plot_imp_path}")
print("=" * 70)
