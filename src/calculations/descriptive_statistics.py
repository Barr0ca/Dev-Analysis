import numpy as np

# Função para calcular a média truncada (trimmed mean), removendo os valores extremos (10% de cada extremidade)
def trimmed_mean(values, trim=0.1):
  sorted = np.sort(values) 
  n = len(sorted)
  cut = int(np.floor(trim * n))
  if 2 * cut >= n:
      raise ValueError("Trim percentage is too large for the dataset size.")
  return np.mean(sorted[cut : n - cut])