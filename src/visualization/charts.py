import pandas as pd
import matplotlib.pyplot as plt

def data_standardization_comparison(data: pd.DataFrame, data_scaled: pd.DataFrame):
  # Visualização dos dados
  fig, axes = plt.subplots(1, 2, figsize=(12, 4))

  # Gráfico antes da padronização
  axes[0].scatter(data["Age"], data["Mental_Health_Index"])
  axes[0].set_title("Before preprocessing")
  axes[0].set_xlabel("Age")
  axes[0].set_ylabel("Mental Health Index")

  # Gráfico após a padronização
  axes[1].scatter(data_scaled["Age"], data_scaled["Mental_Health_Index"])
  axes[1].set_title("After standardization")
  axes[1].set_xlabel("Standardized age")
  axes[1].set_ylabel("Standardized Mental Health Index")

  # Imagem da comparação antes e após a padronização dos dados
  plt.savefig("./reports/figures/comparison.png")