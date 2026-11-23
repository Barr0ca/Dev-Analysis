from src.classifiers.knn import knn_classifier
from src.classifiers.mlp import mlp_classifier
from src.classifiers.naive_bayes import nb_classifier
from src.classifiers.svm import svm_classifier
from src.classifiers.tree_decision import tree_classifier
from src.clustering.grouping import run_grouping
from src.data.load_data import load_data
from src.preprocessing.clean_data import null_values, select_numeric
from src.preprocessing.standardize_data import standardize_numeric_data
from src.visualization.charts import (
    classification_charts,
    data_standardization_comparison,
    knn_charts,
    rna_svm_charts,
)


def main():
    # Carrega os dados do arquivo CSV
    data = load_data("data/raw/dataset.csv")
    print("Dados carregados com sucesso")

    # Seleciona os dados numéricos
    data_numeric = select_numeric(data)
    print("Dados numéricos selecionados com sucesso")

    # Preenche os valores nulos com a média da coluna (filtrada pela idade)
    data_null = null_values(data_numeric)
    print("Valores nulos preenchidos com sucesso")

    # Padroniza os dados numéricos
    data_scaled = standardize_numeric_data(data_null)
    print("Dados padronizados com sucesso")

    # Compara os dados antes e depois da padronização
    data_standardization_comparison(data, data_scaled)
    print("Comparação dos dados antes e depois da padronização realizada com sucesso")

    # Executa o agrupamento e calcula as métricas
    (
        results,
        comparison,
        profile,
        kmeans_label_comparison,
        hierarchical_label_comparison,
    ) = run_grouping(data)

    # Executa o classificador Naive Bayes
    nb_result = nb_classifier(data)
    print("\nClassificador Naive Bayes executado com sucesso")

    # Executa o classificador de árvore de decisão
    tree_result = tree_classifier(data)
    print("\nClassificador de árvore de decisão executado com sucesso")

    # Gráficos da etapa de classificação
    classification_charts(
        data,
        nb_result["scores"],
        tree_result["scores"],
        nb_result["comparison"],
        tree_result["comparison"],
    )
    print("Gráficos da classificação gravados com sucesso")

    # Executa o classificador k-NN (k escolhido por GridSearchCV)
    knn_result = knn_classifier(data)
    print("\nClassificador k-NN executado com sucesso")

    # Gráficos do k-NN e comparação com os demais classificadores
    knn_charts(
        knn_result["k_results"],
        knn_result["best_k"],
        nb_result["scores"],
        tree_result["scores"],
        knn_result["scores"],
        knn_result["comparison"],
    )
    print("Gráficos do k-NN gravados com sucesso")

    # Executa a rede neural MLP (hiperparâmetros escolhidos por GridSearchCV)
    mlp_result = mlp_classifier(data)
    print("\nClassificador MLP executado com sucesso")

    # Executa o SVM (hiperparâmetros escolhidos por GridSearchCV)
    svm_result = svm_classifier(data)
    print("\nClassificador SVM executado com sucesso")

    # Gráficos do MLP e do SVM e comparação entre os cinco classificadores
    rna_svm_charts(
        {
            "Naive Bayes": nb_result["scores"],
            "Árvore de decisão": tree_result["scores"],
            "k-NN": knn_result["scores"],
            "MLP": mlp_result["scores"],
            "SVM": svm_result["scores"],
        },
        mlp_result["loss_curve"],
        mlp_result["validation_scores"],
        mlp_result["comparison"],
        svm_result["comparison"],
    )
    print("Gráficos do MLP e do SVM gravados com sucesso")

    print("\nResultados para escolha de k:")
    print(results)

    print("\nComparação entre algoritmos:")
    print(comparison)

    print("\nPerfil dos clusters:")
    print(profile)

    print("\nK-Means - Clusters versus Overall_Impact:")
    print(kmeans_label_comparison)

    print("\nHierárquico - Clusters versus Overall_Impact:")
    print(hierarchical_label_comparison)


if __name__ == "__main__":
    main()
