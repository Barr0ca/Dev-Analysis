# Impacto das redes sociais na saúde mental de estudantes

Projeto da disciplina de Análise de Dados, desenvolvido em dupla por **Ian de Araújo** e **Jardson Alan**.

## Objetivo

Este estudo pretende analisar dados de 4.500 estudantes, do ensino médio à pós-graduação, para investigar se o maior tempo diário de uso de redes sociais está associado a um menor índice de saúde mental. Busca-se medir a intensidade dessa relação e verificar se ela se mantém ao considerar fatores como idade, gênero e nível acadêmico.

O atributo alvo é o `Mental_Health_Index` (índice de saúde mental). Ele será analisado em função do tempo diário de uso de redes sociais (`Daily_Usage_Hours`), que é o principal atributo explicativo. Idade (`Age`), gênero (`Gender`) e nível acadêmico (`Academic_Level`) entram como fatores de controle.

## Conjunto de dados

O arquivo `data/raw/dataset.csv` reúne respostas de **4.500 estudantes**, do ensino médio à pós-graduação, sobre tempo de tela, sono, estresse e desempenho acadêmico. São **16 atributos** demográficos, comportamentais e psicométricos. A fonte é o dataset [Impact of Social Media on Life](https://www.kaggle.com/datasets/harishyadav0506/impact-of-social-media-on-life), publicado por Harish Yadav no Kaggle sob licença CC0 (domínio público). No Kaggle o arquivo se chama `Social_media_impact_on_life.csv`.

No arquivo local não há valores ausentes em `Mental_Health_Index` nem em `Daily_Usage_Hours`. Há ausências em `Perceived_Stress_Score` (46 registros) e em `Academic_Performance_GPA` (85 registros).

### Atributos

| Atributo | Tipo | Descrição |
| --- | --- | --- |
| `Student_ID` | Texto | Identificador único do estudante |
| `Age` | Inteiro | Idade em anos completos (15 a 26) |
| `Gender` | Categórico | Gênero informado: Female, Male, Non-Binary, Prefer not to say |
| `Academic_Level` | Categórico | Nível acadêmico: High School, Undergraduate ou Postgraduate |
| `Primary_Platform` | Categórico | Plataforma mais usada: Instagram, TikTok, YouTube, Snapchat, X (Twitter), Reddit ou LinkedIn |
| `Daily_Usage_Hours` | Numérico | Tempo médio diário em redes sociais, em horas. Atributo explicativo principal |
| `Weekend_Extra_Hours` | Numérico | Horas extras de tela nos fins de semana, em relação aos dias de semana |
| `Device_Type` | Categórico | Dispositivo principal: Smartphone, Laptop/PC ou Tablet |
| `Sleep_Duration_Hours` | Numérico | Duração média do sono por noite, em horas |
| `Sleep_Quality_Score` | Inteiro | Qualidade do sono autodeclarada, de 1 (muito ruim) a 5 (excelente) |
| `Late_Night_Usage` | Booleano | Uso frequente de tela depois da meia-noite |
| `Social_Comparison_Frequency` | Categórico | Frequência de comparação social: Never, Rarely, Sometimes, Frequently ou Always |
| `Perceived_Stress_Score` | Numérico | Estresse percebido na escala PSS, de 0 a 40 |
| `Mental_Health_Index` | Inteiro | Índice composto de bem-estar. No arquivo, de 32 a 98. **Atributo alvo** |
| `Academic_Performance_GPA` | Numérico | GPA acumulado. No arquivo, de 1,9 a 4,0 |
| `Overall_Impact` | Categórico | Classificação do impacto geral: Beneficial, Neutral ou Negative |

## Pré-processamento

Registro de 23 de outubro de 2026. O fluxo desta etapa é o `main.py`, sobre `data/raw/dataset.csv`.

A leitura fica em `src/data/load_data.py` e devolve a tabela inteira. As 4.500 linhas permanecem, porque a ausência está em células de duas colunas: 46 registros em `Perceived_Stress_Score` e 85 em `Academic_Performance_GPA`. `Mental_Health_Index` e `Daily_Usage_Hours` seguem com os valores observados.

`src/preprocessing/clean_data.py` mantém as colunas numéricas: `Age`, `Daily_Usage_Hours`, `Weekend_Extra_Hours`, `Sleep_Duration_Hours`, `Sleep_Quality_Score`, `Perceived_Stress_Score`, `Mental_Health_Index` e `Academic_Performance_GPA`. `Student_ID` e os atributos categóricos ou booleanos ficam fora desta etapa. Média e padronização se aplicam às medidas numéricas. `Age` permanece na tabela e define o grupo do preenchimento.

Cada vazio de estresse e de GPA recebe a média truncada de 10% em cada extremidade, calculada em `src/calculations/descriptive_statistics.py` entre os estudantes da mesma idade. O preenchimento grava esse valor só na célula vazia.

As colunas numéricas estão em escalas diferentes: idade de 15 a 26 anos, horas de uso, de sono e de fim de semana, qualidade do sono de 1 a 5, estresse de 0 a 40, índice de saúde mental de 32 a 98 e GPA de 1,9 a 4,0. `src/preprocessing/standardize_data.py` aplica `StandardScaler`. Cada coluna numérica passa a ter média 0 e desvio-padrão 1.

`src/visualization/charts.py` grava `reports/figures/comparison.png`. A figura compara `Age` e `Mental_Health_Index` antes e depois da padronização: a nuvem de pontos conserva o desenho, e os eixos passam a estar em desvios-padrão.

Ordem executada por `python main.py`, na raiz do projeto:

1. Carregar `data/raw/dataset.csv`
2. Selecionar as colunas numéricas
3. Preencher ausentes com a média truncada por idade
4. Padronizar com `StandardScaler`
5. Salvar a comparação de `Age` e `Mental_Health_Index`
