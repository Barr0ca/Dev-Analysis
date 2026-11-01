import pandas as pd

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
  silhouette_score,
  calinski_harabasz_score,
  davies_bouldin_score,
  adjusted_rand_score,
  normalized_mutual_info_score,
)
from sklearn.preprocessing import LabelEncoder

from src.preprocessing.clean_data import null_values
from src.preprocessing.standardize_data import standardize_numeric_data

# Define as colunas que serão utilizadas para o agrupamento
FEATURES = [
  "Age",
  "Daily_Usage_Hours",
  "Weekend_Extra_Hours",
  "Sleep_Duration_Hours",
  "Sleep_Quality_Score",
  "Perceived_Stress_Score",
  "Academic_Performance_GPA",
]

# Prepara os dados para o agrupamento
def prepare_data(data: pd.DataFrame):
  features = data[FEATURES].copy()
  features = null_values(features)
  features_scaled = standardize_numeric_data(features)

  labels = LabelEncoder().fit_transform(
    data["Overall_Impact"]
  )

  return features, features_scaled, labels

# Calcula as métricas de avaliação do agrupamento
def calculate_metrics(data, clusters, labels):
  return {
    "silhouette": silhouette_score(data, clusters),
    "calinski_harabasz": calinski_harabasz_score(
      data,
      clusters,
    ),
    "davies_bouldin": davies_bouldin_score(
      data,
      clusters,
    ),
    "adjusted_rand": adjusted_rand_score(
      labels,
      clusters,
    ),
    "normalized_mutual_info": normalized_mutual_info_score(
      labels,
      clusters,
    ),
  }

# Executa o agrupamento e calcula as métricas
def run_grouping(data: pd.DataFrame):
  features, features_scaled, labels = prepare_data(data)

  results = []

  for number_of_clusters in range(2, 7):
    kmeans = KMeans(
      n_clusters=number_of_clusters,
      init="k-means++",
      n_init=20,
      random_state=42,
    )

    kmeans_clusters = kmeans.fit_predict(features_scaled)

    results.append({
      "algorithm": "K-Means",
      "k": number_of_clusters,
      **calculate_metrics(
        features_scaled,
        kmeans_clusters,
        labels,
      ),
    })

  results_table = pd.DataFrame(results)

  best_k = int(
    results_table.loc[
      results_table["silhouette"].idxmax(),
      "k",
    ]
  )

  kmeans = KMeans(
    n_clusters=best_k,
    init="k-means++",
    n_init=50,
    max_iter=500,
    random_state=42,
  )

  kmeans_clusters = kmeans.fit_predict(features_scaled)

  hierarchical = AgglomerativeClustering(
    n_clusters=best_k,
    linkage="ward",
  )

  hierarchical_clusters = hierarchical.fit_predict(
    features_scaled
  )

  comparison = pd.DataFrame({
    "K-Means": calculate_metrics(
      features_scaled,
      kmeans_clusters,
      labels,
    ),
    "Hierárquico": calculate_metrics(
      features_scaled,
      hierarchical_clusters,
      labels,
    ),
  }).T

  profile = features.copy()
  profile["KMeans_Cluster"] = kmeans_clusters

  cluster_profile = profile.groupby(
    "KMeans_Cluster"
  )[FEATURES].mean()

  kmeans_label_comparison = pd.crosstab(
    data["Overall_Impact"],
    kmeans_clusters,
    rownames=["Overall_Impact"],
    colnames=["Cluster K-Means"],
  )

  hierarchical_label_comparison = pd.crosstab(
    data["Overall_Impact"],
    hierarchical_clusters,
    rownames=["Overall_Impact"],
    colnames=["Cluster Hierárquico"],
  )

  return (
    results_table,
    comparison,
    cluster_profile,
    kmeans_label_comparison,
    hierarchical_label_comparison,
  )