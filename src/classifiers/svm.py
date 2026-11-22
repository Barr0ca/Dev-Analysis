import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
  accuracy_score,
  balanced_accuracy_score,
  classification_report,
  roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC

from src.clustering.grouping import prepare_data

def svm_classifier(data: pd.DataFrame):
  # Entradas sem padronizar: o SVM usa distância e produto interno, então a
  # padronização fica dentro do Pipeline e é aprendida só no treino de cada parte
  X, _, y = prepare_data(data)

  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=125
  )

  # SVM (máquina de vetores de suporte). A mesma divisão dos outros classificadores
  # deixa a comparação reproduzível.
  pipeline = Pipeline(
    [
      ("scaler", StandardScaler()),
      ("model", SVC(random_state=125)),
    ]
  )

  # A grade escolhe o kernel e os parâmetros só no recorte de treino. O gamma só tem
  # efeito no kernel rbf, por isso o kernel linear tem uma grade própria.
  search = GridSearchCV(
    pipeline,
    [
      {
        "model__kernel": ["linear"],
        "model__C": [0.1, 1, 10, 100],
        "model__class_weight": [None, "balanced"],
      },
      {
        "model__kernel": ["rbf"],
        "model__C": [0.1, 1, 10, 100],
        "model__gamma": ["scale", 0.01, 0.1, 1],
        "model__class_weight": [None, "balanced"],
      },
    ],
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
  )
  search.fit(X_train, y_train)
  model = search.best_estimator_

  print("\nSVM (máquina de vetores de suporte)")
  print("Melhores hiperparâmetros:", search.best_params_)
  print("Acurácia média da busca (melhor conjunto):", search.best_score_)

  # Desempenho pela validação cruzada, com os hiperparâmetros escolhidos
  scores = cross_val_score(
    clone(pipeline).set_params(**search.best_params_),
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
  print("Acurácia balanceada no teste:", balanced_accuracy_score(y_test, predicted))

  # AUC multiclasse (um contra os demais, média macro), calculada com a distância
  # à fronteira de decisão, pois o SVC não gera probabilidades sem custo extra
  auc = roc_auc_score(
    label_binarize(y_test, classes=model.classes_),
    model.decision_function(X_test),
    average="macro",
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

  return {
    "model": model,
    "scores": scores,
    "comparison": comparison,
    "best_params": search.best_params_,
  }
