import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Função para calcular a média truncada (trimmed mean), removendo os valores extremos (10% de cada extremidade)
def trimmed_mean(values, trim=0.1):
    sorted = np.sort(values) 
    n = len(sorted)
    cut = int(np.floor(trim * n))
    if 2 * cut >= n:
        raise ValueError("Trim percentage is too large for the dataset size.")
    return np.mean(sorted[cut : n - cut])

# Base de dados
data = pd.read_csv("DataSet.csv")

# Identificação de valores nulos ou duplicados
# print(data.isnull().sum()) # Verifica se há valores nulos
# print(data.duplicated().sum()) # Verifica se há valores duplicados

# Selecionando dos dados numericos
data_numeric = data.select_dtypes(include="number")

# Preenchendo valores nulos com a média da coluna (filtrada pela idade)
cols = ["Perceived_Stress_Score", "Academic_Performance_GPA"]
data_numeric[cols] = data_numeric[cols].fillna(
    data_numeric.groupby("Age")[cols].transform(trimmed_mean)
)

# Identificou-se que as colunas "Perceived_Stress_Score" e "Academic_Performance_GPA" possuem, respectivamente, 46 e 85 valores nulos.
# decidiu-se preencher esses valores com a média da coluna em questão filtrando pela idade.

# Padronização dos dados numericos
scaler = StandardScaler()
data_scaled = pd.DataFrame(
    scaler.fit_transform(data_numeric), columns=data_numeric.columns
)

# Visualização dos dados
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Gráfico antes da padronização
axes[0].scatter(data_numeric["Age"], data_numeric["Mental_Health_Index"])
axes[0].set_title("Before preprocessing")
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Mental Health Index")

# Gráfico após a padronização
axes[1].scatter(data_scaled["Age"], data_scaled["Mental_Health_Index"])
axes[1].set_title("After standardization")
axes[1].set_xlabel("Standardized age")
axes[1].set_ylabel("Standardized Mental Health Index")

# Imagem da comparação antes e após a padronização dos dados
plt.savefig("./pre-processing/images/comparison.png")


