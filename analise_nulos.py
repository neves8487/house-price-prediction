import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregar os dados de treino e de teste
train_df = pd.read_csv(os.path.join("data", "train.csv"))
test_df = pd.read_csv(os.path.join("data", "test.csv"))

print("=" * 70)
print("ANÁLISE DE VALORES EM FALTA (MISSING VALUES)")
print("=" * 70)

# 2. Calcular valores nulos no treino
total_linhas = len(train_df)
nulos_treino = train_df.isnull().sum()
nulos_treino = nulos_treino[nulos_treino > 0].sort_values(ascending=False)

nulos_df = pd.DataFrame({
    'Quantidade_Nulos': nulos_treino,
    'Percentagem': (nulos_treino / total_linhas) * 100,
    'Tipo': [train_df[col].dtype for col in nulos_treino.index]
})

print(f"\nExistem {len(nulos_df)} colunas com valores em falta no conjunto de treino (de um total de {train_df.shape[1]} colunas).\n")
print(f"{'Coluna':<18} | {'Nulos':<7} | {'%':<6} | {'Tipo':<10} | {'Significado no Dicionário'}")
print("-" * 75)

# Dicionário com o significado dos NAs segundo o data_description.txt
significados = {
    'PoolQC': 'Sem Piscina ("No Pool")',
    'MiscFeature': 'Sem recurso extra ("None")',
    'Alley': 'Sem acesso a beco ("No alley access")',
    'Fence': 'Sem vedação/cerca ("No Fence")',
    'MasVnrType': 'Sem revestimento de alvenaria ("None")',
    'FireplaceQu': 'Sem lareira ("No Fireplace")',
    'LotFrontage': 'Frente de rua não medida (Verdadeiro Nulo)',
    'GarageType': 'Sem garagem ("No Garage")',
    'GarageYrBlt': 'Sem garagem (Ano inexistente)',
    'GarageFinish': 'Sem garagem ("No Garage")',
    'GarageQual': 'Sem garagem ("No Garage")',
    'GarageCond': 'Sem garagem ("No Garage")',
    'BsmtExposure': 'Sem cave ("No Basement")',
    'BsmtFinType2': 'Sem cave ("No Basement")',
    'BsmtFinType1': 'Sem cave ("No Basement")',
    'BsmtCond': 'Sem cave ("No Basement")',
    'BsmtQual': 'Sem cave ("No Basement")',
    'MasVnrArea': 'Área de alvenaria (0 sq ft)',
    'Electrical': 'Sistema elétrico em falta (1 casa no treino)'
}

for col, row in nulos_df.iterrows():
    sig = significados.get(col, "Verificar no dicionário")
    print(f"{col:<18} | {int(row['Quantidade_Nulos']):<7} | {row['Percentagem']:>5.1f}% | {str(row['Tipo']):<10} | {sig}")

# 3. Gerar gráfico visual
os.makedirs("plots", exist_ok=True)
output_img = os.path.join("plots", "analise_nulos.png")

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(12, 7))

# Gráfico de barras horizontais
sns.barplot(
    x=nulos_df['Percentagem'], 
    y=nulos_df.index, 
    ax=ax, 
    hue=nulos_df.index,
    palette="flare_r",
    legend=False
)

ax.set_title("Percentagem de Valores em Falta por Coluna (Conjunto de Treino)", fontsize=14, fontweight='bold')
ax.set_xlabel("% de Linhas Vazias", fontsize=11)
ax.set_ylabel("Nome da Característica", fontsize=11)

# Adicionar rótulos numéricos em cada barra
for i, v in enumerate(nulos_df['Percentagem']):
    ax.text(v + 0.5, i, f"{v:.1f}%", va='center', fontsize=9, fontweight='bold', color='#333333')

plt.xlim(0, 105)
plt.tight_layout()
plt.savefig(output_img, dpi=150)
plt.close()

print("\n" + "=" * 70)
print(f"Gráfico guardado em: {output_img}")
print("=" * 70)
