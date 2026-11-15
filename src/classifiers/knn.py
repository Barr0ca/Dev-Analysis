import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from src.clustering.grouping import prepare_data


def knn_classifier(data: pd.DataFrame):
  # Entradas já padronizadas: o k-NN é sensível à escala por usar distância
  X, _, y = prepare_data(data)

  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=125
  )

  # k-NN. A mesma divisão dos outros classificadores deixa a comparação reproduzível.
  # A grade escolhe o k só no recorte de treino. Valores ímpares evitam empates na votação.
  search = GridSearchCV(
    KNeighborsClassifier(),
    {"n_neighbors": list(range(1, 32, 2))},
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
  )
  search.fit(X_train, y_train)
  model = search.best_estimator_

  print("\nk-NN (k vizinhos mais próximos)")
  print("Melhores hiperparâmetros:", search.best_params_)
  print("Acurácia média da busca (melhor k):", search.best_score_)

  # Desempenho pela validação cruzada, com o k escolhido
  scores = cross_val_score(
    KNeighborsClassifier(**search.best_params_),
    X,
    y,
    cv=5,
    scoring="accuracy",
  )
  print("Validação cruzada (5 folds):", scores)
  print("Acurácia média:", scores.mean())
  print("Desvio-padrão:", scores.std())

  encoder = LabelEncoder()
  encoder.fit(data["Overall_Impact"])

  predicted = model.predict(X_test)
  real_labels = encoder.inverse_transform(y_test)
  predicted_labels = encoder.inverse_transform(predicted)

  print("Acurácia no teste:", accuracy_score(y_test, predicted))

  # AUC multiclasse (um contra os demais, média macro)
  auc = roc_auc_score(
    y_test, model.predict_proba(X_test), multi_class="ovr", average="macro"
  )
  print("ROC AUC no teste (OvR, macro):", auc)

  print("\nRelatório por classe:")
  print(
    classification_report(
      y_test,
      predicted,
      target_names=encoder.classes_,
    )
  )

  comparison = pd.crosstab(
    real_labels,
    predicted_labels,
    rownames=["Real"],
    colnames=["Previsto"],
  )
  print("\nMatriz de confusão:")
  print(comparison)

  # Acurácia média de cada k na busca (insumo do gráfico de impacto do k)
  k_results = pd.DataFrame(
    {
      "k": search.cv_results_["param_n_neighbors"].astype(int),
      "accuracy": search.cv_results_["mean_test_score"],
      "std": search.cv_results_["std_test_score"],
    }
  )

  return {
    "model": model,
    "scores": scores,
    "comparison": comparison,
    "k_results": k_results,
    "best_k": int(search.best_params_["n_neighbors"]),
  }
