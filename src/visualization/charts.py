import matplotlib.pyplot as plt
import pandas as pd


def data_standardization_comparison(data: pd.DataFrame, data_scaled: pd.DataFrame):
    # Visualização dos dados
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Gráfico antes da padronização
    axes[0].scatter(data["Age"], data["Mental_Health_Index"])
    axes[0].set_title("Before preprocessing")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Mental Health Index")

    # Gráfico após a padronização
    axes[1].scatter(data_scaled["Age"], data_scaled["Mental_Health_Index"])
    axes[1].set_title("After standardization")
    axes[1].set_xlabel("Standardized age")
    axes[1].set_ylabel("Standardized Mental Health Index")

    # Imagem da comparação antes e após a padronização dos dados
    plt.savefig("./reports/figures/comparison.png")
    plt.close(fig)


def classification_charts(data, nb_scores, tree_scores, nb_matrix, tree_matrix):
    # Contagem das classes de Overall_Impact
    order = ["Beneficial", "Neutral", "Negative"]
    counts = data["Overall_Impact"].value_counts().reindex(order)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(order, counts.values, color="#4C78A8")
    ax.bar_label(bars, padding=3)
    ax.set_ylim(0, counts.max() * 1.12)
    ax.set_title("Classes de Overall Impact")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Registros")
    fig.tight_layout()
    fig.savefig("./reports/figures/class_counts.png", dpi=150)
    plt.close(fig)

    # Acurácia das cinco partes da validação cruzada
    parts = ["1", "2", "3", "4", "5"]
    positions = range(len(parts))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    nb_bars = ax.bar(
        [position - width / 2 for position in positions],
        nb_scores,
        width,
        label="Naive Bayes",
        color="#4C78A8",
    )
    tree_bars = ax.bar(
        [position + width / 2 for position in positions],
        tree_scores,
        width,
        label="Árvore de decisão",
        color="#F58518",
    )
    ax.bar_label(nb_bars, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(tree_bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(parts)
    ax.set_ylim(0.85, 1.02)
    ax.set_title("Validação cruzada")
    ax.set_xlabel("Parte")
    ax.set_ylabel("Acurácia")
    ax.legend()
    fig.tight_layout()
    fig.savefig("./reports/figures/cross_validation.png", dpi=150)
    plt.close(fig)

    # Matrizes de confusão do recorte de teste
    labels = ["Beneficial", "Negative", "Neutral"]
    matrices = (
        (nb_matrix.reindex(index=labels, columns=labels), "Naive Bayes"),
        (tree_matrix.reindex(index=labels, columns=labels), "Árvore de decisão"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (matrix, title) in zip(axes, matrices):
        values = matrix.to_numpy()
        ax.imshow(values, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Previsto")
        ax.set_ylabel("Real")
        ax.set_title(title)
        darkest = values.max() / 2
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                color = "white" if values[row, column] > darkest else "black"
                ax.text(
                    column,
                    row,
                    int(values[row, column]),
                    ha="center",
                    va="center",
                    color=color,
                )
    fig.tight_layout()
    fig.savefig("./reports/figures/confusion_matrices.png", dpi=150)
    plt.close(fig)
