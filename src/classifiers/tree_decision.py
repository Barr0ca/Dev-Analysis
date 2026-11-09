import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from src.clustering.grouping import prepare_data


def tree_classifier(data: pd.DataFrame):
    X, _, y = prepare_data(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=125
    )

    # Árvore de decisão. A mesma semente do Naive Bayes deixa a comparação reproduzível.
    # A grade escolhe os hiperparâmetros só no recorte de treino.
    search = GridSearchCV(
        DecisionTreeClassifier(random_state=125),
        {
            "criterion": ["gini", "entropy"],
            "max_depth": [3, 5, 7, 10, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_

    print("\nÁrvore de decisão")
    print("Melhores hiperparâmetros:", search.best_params_)

    # Desempenho pela validação cruzada, com os hiperparâmetros escolhidos
    scores = cross_val_score(
        DecisionTreeClassifier(random_state=125, **search.best_params_),
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

    print("\nRelatório por classe:")
    print(classification_report(
        y_test,
        predicted,
        target_names=encoder.classes_,
    ))

    comparison = pd.crosstab(
        real_labels,
        predicted_labels,
        rownames=["Real"],
        colnames=["Previsto"],
    )
    print("\nMatriz de confusão:")
    print(comparison)

    print("\nExemplo, linha 7:")
    print("Valor real:", real_labels[7])
    print("Valor previsto:", predicted_labels[7])

    return {
        "model": model,
        "scores": scores,
        "comparison": comparison,
    }
