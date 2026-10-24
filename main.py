from src.data.load_data import load_data
from src.preprocessing.clean_data import select_numeric, null_values
from src.preprocessing.standardize_data import standardize_numeric_data
from src.visualization.charts import data_standardization_comparison

def main():
  # Carrega os dados do arquivo CSV
  data = load_data("data/raw/dataset.csv")
  print("Dados carregados com sucesso")

  # Seleciona os dados numéricos
  data_numeric = select_numeric(data)
  print("Dados numéricos selecionados com sucesso")

  # Preenche os valores nulos com a média da coluna (filtrada pela idade)
  data_null = null_values(data_numeric)
  print("Valores nulos preenchidos com sucesso")

  # Padroniza os dados numéricos
  data_scaled = standardize_numeric_data(data_null)
  print("Dados padronizados com sucesso")

  # Compara os dados antes e depois da padronização
  data_standardization_comparison(data, data_scaled)
  print("Comparação dos dados antes e depois da padronização realizada com sucesso")

if __name__ == "__main__":
  main()