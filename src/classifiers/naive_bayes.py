import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

from src.clustering.grouping import prepare_data


def nb_classifier(data: pd.DataFrame):
    X, _, y = prepare_data(  # "_" indica que a segunda coluna nao sera utilizada
        data
    )  # Importacao de "grouping.py" (dados numericos a serem classificados)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=125
    )

    # Classificador GaussianNB
    model = GaussianNB()

    # Modelo de treinamento
    model.fit(X_train, y_train)

    # Codificacao dos valores reais (nomes)
    encoder = LabelEncoder()
    encoder.fit(data["Overall_Impact"])

    predicted = model.predict(X_test)
    real_labels = encoder.inverse_transform(y_test)
    predicted_labels = encoder.inverse_transform(predicted)

    print("\nAcurácia:", accuracy_score(y_test, predicted))

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
