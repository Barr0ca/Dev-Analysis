import pandas as pd
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Padronização dos dados numericos
def standardize_numeric_data(data: pd.DataFrame):
  data_scaled = pd.DataFrame(
    scaler.fit_transform(data), columns=data.columns
  )
  return data_scaled