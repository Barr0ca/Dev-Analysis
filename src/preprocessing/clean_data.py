import pandas as pd

from src.calculations.descriptive_statistics import (
  trimmed_mean
)

# Seleciona dados númericos
def select_numeric(data: pd.DataFrame):
  return data.select_dtypes(include="number")

# Preenchendo valores nulos com a média da coluna (filtrada pela idade)
def null_values(data: pd.DataFrame):
  cols = ["Perceived_Stress_Score", "Academic_Performance_GPA"]
  data = data.copy()
  data[cols] = data[cols].fillna(
    data.groupby("Age")[cols].transform(trimmed_mean)
  )
  return data