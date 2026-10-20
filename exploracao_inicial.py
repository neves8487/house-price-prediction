import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregar os dados de treino
data_path = os.path.join("data", "train.csv")
df = pd.read_csv(data_path)

print("=" * 60)
print(f"DADOS CARREGADOS COM SUCESSO!")
print(f"Total de casas no conjunto de treino: {df.shape[0]}")
print(f"Total de colunas (características): {df.shape[1]}")
print("=" * 60)

# 2. Resumo da variável que queremos prever: SalePrice
target = df['SalePrice']
print("\n--- 1. RESUMO DO PREÇO DAS CASAS (SalePrice) ---")
print(f"Preço Mínimo:  ${target.min():,}")
print(f"Preço Mediano: ${target.median():,}")
print(f"Preço Médio:   ${target.mean():,.2f}")
print(f"Preço Máximo:  ${target.max():,}")
print(f"Desvio Padrão: ${target.std():,.2f}")

# 3. Calcular a Correlação de todas as variáveis numéricas com SalePrice
numeric_df = df.select_dtypes(include=[np.number])
correlacoes = numeric_df.corr()['SalePrice'].sort_values(ascending=False)

print("\n--- 2. TOP 10 VARIÁVEIS COM MAIOR LIGAÇÃO COM O PREÇO ---")
print("Nota: 1.0 é correlação perfeita; acima de 0.60 é considerada uma ligação forte.")
top_10 = correlacoes.iloc[1:11] # excluindo o próprio SalePrice que é 1.0
for rank, (col, corr_val) in enumerate(top_10.items(), 1):
    print(f" {rank:2d}. {col:<15} -> Correlação: {corr_val:+.3f}")

# 4. Criar gráficos explicativos
os.makedirs("plots", exist_ok=True)
output_img = os.path.join("plots", "exploracao_inicial.png")

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle("Análise Exploratória Inicial: Como as Características se Ligam ao Preço", fontsize=16, fontweight='bold')

# Gráfico 1: Distribuição do Preço
sns.histplot(df['SalePrice'], kde=True, ax=axes[0, 0], color='#2b5c8f')
axes[0, 0].set_title("1. Distribuição de SalePrice (Nota a cauda à direita / 'assimetria')", fontsize=12)
axes[0, 0].set_xlabel("Preço de Venda ($)")
axes[0, 0].set_ylabel("Frequência")

# Gráfico 2: Top 10 Correlações
sns.barplot(x=top_10.values, y=top_10.index, ax=axes[0, 1], hue=top_10.index, palette="crest", legend=False)
axes[0, 1].set_title("2. Top 10 Características com Maior Correlação Linear", fontsize=12)
axes[0, 1].set_xlabel("Coeficiente de Correlação (Pearson)")

# Gráfico 3: Área Habitável (GrLivArea) vs Preço
sns.scatterplot(data=df, x='GrLivArea', y='SalePrice', ax=axes[1, 0], alpha=0.6, color='#e76f51')
axes[1, 0].set_title("3. Área Habitável (GrLivArea) vs Preço (Dispersão)", fontsize=12)
axes[1, 0].set_xlabel("Área Útil Acima do Solo (pés quadrados)")
axes[1, 0].set_ylabel("Preço de Venda ($)")

# Gráfico 4: Qualidade Geral (OverallQual) vs Preço
sns.boxplot(data=df, x='OverallQual', y='SalePrice', ax=axes[1, 1], hue='OverallQual', palette="Blues", legend=False)
axes[1, 1].set_title("4. Qualidade Geral dos Materiais (OverallQual: 1 a 10) vs Preço", fontsize=12)
axes[1, 1].set_xlabel("Nota de Qualidade Geral (1 a 10)")
axes[1, 1].set_ylabel("Preço de Venda ($)")

plt.tight_layout()
plt.savefig(output_img, dpi=150)
plt.close()

print("\n" + "=" * 60)
print(f"Gráfico gerado e guardado com sucesso em: {output_img}")
print("=" * 60)
