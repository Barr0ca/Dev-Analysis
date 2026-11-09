import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from src.clustering.grouping import prepare_data


def tree_classifier(data: pd.DataFrame):
    X, _, y = prepare_data(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=125
    )

    # Árvore de decisão. A mesma semente do Naive Bayes deixa a comparação reproduzível.
    model = DecisionTreeClassifier(random_state=125)

    model.fit(X_train, y_train)

    encoder = LabelEncoder()
    encoder.fit(data["Overall_Impact"])

    predicted = model.predict(X_test)
    real_labels = encoder.inverse_transform(y_test)
    predicted_labels = encoder.inverse_transform(predicted)

    print("\nÁrvore de decisão")
    print("Acurácia:", accuracy_score(y_test, predicted))

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

    return model
