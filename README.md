# Impacto das redes sociais na saúde mental de estudantes

Projeto da disciplina de Análise de Dados, desenvolvido em dupla por **Ian de Araújo** e **Jardson Alan**.

O estudo usa respostas de 4.500 estudantes, do ensino médio à pós-graduação, para investigar se um maior tempo diário de redes sociais está associado a um menor índice de saúde mental. O atributo alvo é o `Mental_Health_Index`. O principal atributo explicativo é o `Daily_Usage_Hours`. Idade (`Age`), gênero (`Gender`) e nível acadêmico (`Academic_Level`) entram como fatores de controle.

Os dados vêm do dataset [Impact of Social Media on Life](https://www.kaggle.com/datasets/harishyadav0506/impact-of-social-media-on-life), de Harish Yadav, sob licença CC0. No repositório, o arquivo está em `data/raw/dataset.csv`.

O registro passo a passo do que a equipe faz no repositório fica em [`docs/documentation.md`](docs/documentation.md).

## Estrutura de pastas

```text
.
├── data/
│   └── raw/
│       └── dataset.csv          # dados originais
├── docs/
│   └── documentation.md         # registro do trabalho da equipe
├── reports/
│   └── figures/
│       └── comparison.png       # figura gerada pelo script
├── src/
│   ├── calculations/            # estatísticas, como a média truncada
│   ├── data/                    # leitura do CSV
│   ├── preprocessing/           # limpeza e padronização
│   └── visualization/           # gráficos
├── main.py                      # executa o fluxo completo
├── requirements.txt
└── README.md
```

Cada pasta de `src/` tem um `readme.md` com a responsabilidade dos arquivos que ela contém.

## Como executar

É preciso ter Python 3 instalado.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

No Windows, ative o ambiente com `.venv\Scripts\activate`.

O script carrega o CSV, mantém as colunas numéricas, preenche os valores ausentes com a média truncada dentro de cada idade, padroniza essas colunas e salva a comparação em `reports/figures/comparison.png`. A pasta `reports/figures/` precisa existir antes da execução.
