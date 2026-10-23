# Impacto das redes sociais na saúde mental de estudantes

Projeto da disciplina de Análise de Dados, desenvolvido em dupla por **Ian de Araújo** e **Jardson Alan**.

## Objetivo

Este estudo pretende analisar dados de 4.500 estudantes, do ensino médio à pós-graduação, para investigar se o maior tempo diário de uso de redes sociais está associado a um menor índice de saúde mental. Busca-se medir a intensidade dessa relação e verificar se ela se mantém ao considerar fatores como idade, gênero e nível acadêmico.

O atributo alvo é o `Mental_Health_Index` (índice de saúde mental). Ele será analisado em função do tempo diário de uso de redes sociais (`Daily_Usage_Hours`), que é o principal atributo explicativo. Idade (`Age`), gênero (`Gender`) e nível acadêmico (`Academic_Level`) entram como fatores de controle.

## Conjunto de dados

O arquivo `DataSet.csv` reúne respostas de **4.500 estudantes**, do ensino médio à pós-graduação, sobre tempo de tela, sono, estresse e desempenho acadêmico. São **16 atributos** demográficos, comportamentais e psicométricos. A fonte é o dataset [Impact of Social Media on Life](https://www.kaggle.com/datasets/harishyadav0506/impact-of-social-media-on-life), publicado por Harish Yadav no Kaggle sob licença CC0 (domínio público). No Kaggle o arquivo se chama `Social_media_impact_on_life.csv`.

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
| `Mental_Health_Index` | Inteiro | Índice composto de bem-estar, de 10 a 98. **Atributo alvo** |
| `Academic_Performance_GPA` | Numérico | GPA acumulado, de 1,70 a 4,00 |
| `Overall_Impact` | Categórico | Classificação do impacto geral: Beneficial, Neutral ou Negative |
