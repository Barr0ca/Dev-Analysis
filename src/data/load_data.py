import pandas as pd

def load_data(data: str) -> pd.DataFrame:
  data_loading = pd.read_csv(data)
  return data_loading