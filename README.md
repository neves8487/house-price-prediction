# 🏠 House Prices - Advanced Regression Techniques (Kaggle)

Projeto prático de previsão de preços imobiliários baseado na competição do Kaggle [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

O objetivo deste projeto é prever o preço final de venda de cada casa (`SalePrice`) em Ames, Iowa, utilizando 79 variáveis explicativas que descrevem quase todos os aspetos dos imóveis residenciais.

---

## 📌 Estrutura do Projeto

```bash
house_price/
├── data/
│   ├── train.csv               # Dataset bruto de treino (1.460 casas)
│   ├── test.csv                # Dataset bruto de teste (1.459 casas)
│   ├── data_description.txt    # Dicionário de dados oficial
│   ├── train_clean.csv         # Dados de treino pré-processados (zero nulos)
│   └── test_clean.csv          # Dados de teste pré-processados (zero nulos)
├── plots/
│   ├── exploracao_inicial.png  # Gráficos da distribuição e correlações
│   └── analise_nulos.png       # Gráfico com o ranking de missing values
├── exploracao_inicial.py       # Script de Análise Exploratória (EDA)
├── analise_nulos.py            # Script de mapeamento e interpretação de nulos
├── preprocessamento.py         # Pipeline de Feature Engineering e Encoding
├── .gitignore
└── README.md
```

---

## 🔍 1. Análise Exploratória de Dados (EDA)

Na primeira fase, explorámos a distribuição do preço e calculámos o coeficiente de correlação de Pearson para identificar os fatores que mais afetam o valor dos imóveis:

- **Target (`SalePrice`)**: A distribuição original é positivamente assimétrica (*right-skewed*), variando entre **$34.900** e **$755.000** (mediana: **$163.000**). Por esta razão, aplicou-se a transformação logarítmica `np.log1p` para aproximar a distribuição à normalidade e alinhar com a métrica da competição (**RMSLE**).
- **Top Fatores Correlacionados**:
  1. `OverallQual` (+0.791): Qualidade geral dos materiais e acabamentos.
  2. `GrLivArea` (+0.709): Área habitável acima do solo.
  3. `GarageCars` (+0.640) / `GarageArea` (+0.623): Capacidade e dimensão da garagem.
  4. `TotalBsmtSF` (+0.614): Área da cave (*basement*).
  5. `1stFlrSF` (+0.606): Área do piso térreo.
  6. `FullBath` (+0.561): Número de casas de banho completas.
  7. `YearBuilt` (+0.523): Ano de construção original.

---

## 🛠️ 2. Tratamento de Dados e Engenharia de Atributos (*Feature Engineering*)

### A) Tratamento de Falsos Nulos
- Colunas como `PoolQC`, `MiscFeature`, `Alley`, `Fence`, `FireplaceQu`, `GarageType` e `BsmtQual` possuem valores `NA` que representam a **ausência do elemento** (ex: "Sem Piscina", "Sem Garagem"), e não erros de medição. Foram convertidas para `'None'` ou `0`.
- Para o `LotFrontage` (largura da frente da casa), imputámos os valores em falta através da **mediana do respetivo bairro (`Neighborhood`)**.

### B) Mapeamento Ordinal
- Variáveis qualitativas com escala de avaliação explícita (`Ex`, `Gd`, `TA`, `Fa`, `Po`, `None`) foram convertidas numa escala numérica de `5` a `0`.
- A coluna `Fence` (cerca) foi mapeada de `0` a `4` com base no grau de privacidade e material (`GdPrv` = 4, `MnPrv` = 3, `GdWo` = 2, `MnWw` = 1, `None` = 0).

### C) Novas Variáveis Criadas
- `TotalSF`: Área útil total ($\text{Cave} + \text{1º Piso} + \text{2º Piso}$).
- `TotalBath`: Total ponderado de casas de banho ($\text{FullBath} + 0.5 \times \text{HalfBath} + \text{BsmtFullBath} + 0.5 \times \text{BsmtHalfBath}$).
- `HouseAge`: Idade do imóvel no ano da venda ($\text{YrSold} - \text{YearBuilt}$).
- `RemodAge`: Anos desde a última remodelação ($\text{YrSold} - \text{YearRemodAdd}$).
- `TotalPorchSF`: Soma das áreas de varandas e alpendres.
- `OverallScore`: Interação multiplicativa entre qualidade e condição ($\text{OverallQual} \times \text{OverallCond}$).
- Indicadores binários: `HasPool`, `HasGarage`, `HasBasement`, `HasFireplace`, `Has2ndFloor`.

---

## 🚀 Como Executar Localmente

### 1. Criar e ativar o ambiente virtual:
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 2. Instalar as dependências:
```powershell
pip install pandas numpy scikit-learn lightgbm xgboost matplotlib seaborn
```

### 3. Executar os scripts:
```powershell
# Análise exploratória inicial (gera plots/exploracao_inicial.png)
python exploracao_inicial.py

# Análise de missing values (gera plots/analise_nulos.png)
python analise_nulos.py

# Pré-processamento e geração dos ficheiros limpos
python preprocessamento.py
```

---

---

## 🏆 3. Modelação e Resultados (5-Fold Cross-Validation)

Utilizámos uma estratégia de validação cruzada robusta com 5 folds, avaliando com a métrica oficial **RMSLE** (*Root Mean Squared Log Error*):

| Posição | Modelo | RMSLE CV (Menor é Melhor) | Notas |
| :---: | :--- | :---: | :--- |
| **🥇 #1** | **ENSEMBLE (Blend Ponderado)** | **0.1115** | Combinação ponderada de lineares e árvores |
| **🥈 #2** | **Lasso (Linear L1)** | **0.1148** | Seleção esparsa de variáveis com regularização |
| **🥉 #3** | **Ridge (Linear L2)** | **0.1152** | Regularização quadrática suave |
| **#4** | **XGBoost** | **0.1176** | Gradient Boosted Trees (1200 estimators) |
| **#5** | **LightGBM** | **0.1214** | Gradient Boosted Trees rápido |

> **Insight Teórico**: O **Ensemble** bateu todos os modelos individuais isolados, demonstrando o poder de combinar abordagens lineares com modelos baseados em árvores.

---

## 📊 4. Importância das Características (*Feature Importance*)

O gráfico gerado em `plots/importancia_features.png` demonstra que as variáveis de **área total útil (`TotalSF`)**, **qualidade geral (`OverallQual`)**, **área habitável (`GrLivArea`)** e **idade do imóvel (`HouseAge`)** foram as mais decisivas na previsão do preço.

---

## 🚀 Como Executar Localmente

### 1. Criar e ativar o ambiente virtual:
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 2. Instalar as dependências:
```powershell
pip install pandas numpy scikit-learn lightgbm xgboost matplotlib seaborn
```

### 3. Executar o pipeline completo:
```powershell
# 1. Análise exploratória inicial (gera plots/exploracao_inicial.png)
python exploracao_inicial.py

# 2. Análise de missing values (gera plots/analise_nulos.png)
python analise_nulos.py

# 3. Pré-processamento e geração dos ficheiros limpos
python preprocessamento.py

# 4. Treino com 5-Fold CV, Ensemble e geração do submission.csv
python treinar_modelos.py
```

---

## 📤 5. Submissão no Kaggle
O ficheiro final [submission.csv](submission.csv) contém 1.459 previsões prontas para upload na página da competição do Kaggle.
