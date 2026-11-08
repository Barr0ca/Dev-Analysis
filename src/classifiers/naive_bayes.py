import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from src.clustering.grouping import prepare_data


def nb_classifier(data: pd.DataFrame):
    X, _, y = prepare_data( # "_" indica que a segunda coluna nao sera utilizada
        data
    )  # Importacao de "grouping.py" (dados numericos a serem classificados)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=125
    )

    # Classificador GaussianNB
    model = GaussianNB()

    # Modelo de treinamento
    model.fit(X_train, y_train)

    # Predicao
    predicted = model.predict(X_test.iloc[[7]])

    print("\nValor real:", y_test[7])
    print("Valor previsto pelo modelo:", predicted[0])

    return model