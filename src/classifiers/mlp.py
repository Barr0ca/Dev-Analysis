import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
  accuracy_score,
  balanced_accuracy_score,
  classification_report,
  roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.clustering.grouping import prepare_data

def mlp_classifier(data: pd.DataFrame):
  # Entradas sem padronizar: o MLP é sensível à escala, então a padronização
  # fica dentro do Pipeline e é aprendida só nos dados de treino de cada parte
  X, _, y = prepare_data(data)

  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=125
  )

  # Rede neural MLP. A mesma divisão dos outros classificadores deixa a comparação reproduzível.
  # A parada antecipada separa 20% do treino para validação e encerra o treino sem ganho.
  pipeline = Pipeline(
    [
      ("scaler", StandardScaler()),
      (
        "model",
        MLPClassifier(
          solver="adam",
          max_iter=500,
          early_stopping=True,
          validation_fraction=0.20,
          n_iter_no_change=20,
          random_state=125,
        ),
      ),
    ]
  )

  # A grade escolhe a topologia e os parâmetros de treino só no recorte de treino.
  search = GridSearchCV(
    pipeline,
    {
      "model__hidden_layer_sizes": [(32,), (32, 16), (64, 32), (64, 32, 16)],
      "model__activation": ["relu", "tanh"],
      "model__alpha": [0.0001, 0.001, 0.01],
      "model__learning_rate_init": [0.001, 0.01],
    },
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
  )
  search.fit(X_train, y_train)
  model = search.best_estimator_

  print("\nRede neural MLP")
  print("Melhores hiperparâmetros:", search.best_params_)
  print("Acurácia média da busca (melhor conjunto):", search.best_score_)
  print("Iterações realizadas:", model.named_steps["model"].n_iter_)

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

  mlp = model.named_steps["model"]

  return {
    "model": model,
    "scores": scores,
    "comparison": comparison,
    "best_params": search.best_params_,
    "loss_curve": mlp.loss_curve_,
    "validation_scores": mlp.validation_scores_,
  }
