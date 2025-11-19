🦸‍♀️ Super-Heróis – Exploração de Dados e Modelos de Machine Learning

Este projeto foi desenvolvido como parte de um desafio técnico envolvendo análise exploratória, clustering, classificação e regressão usando dados de super-heróis.
A solução inclui também uma aplicação interativa construída em Streamlit, permitindo ao usuário explorar os dados e interagir com os modelos desenvolvidos.

📁 Conteúdo do repositório

app.py – Aplicação Streamlit que integra exploração dos dados, clustering, classificação e regressão.

heroes_information.csv – Arquivo com informações gerais dos super-heróis.

super_hero_powers.csv – Arquivo binário contendo poderes de cada herói.

alelo.ipynb – Notebook utilizado durante a resolução das questões.

🚀 Funcionalidades da Aplicação

A aplicação permite ao usuário:

🔍 1. Explorar os dados

Visualização completa das tabelas

Estatísticas descritivas

Distribuições de variáveis

Filtros por:

Alignment (good / bad / neutral)

Gender

Publisher

Gráficos interativos (barras e histogramas)

🧩 2. Clustering (Agrupamento) – Questão 1

Redução de dimensionalidade com PCA

Agrupamento usando K-Means

Visualização dos clusters em 2D

Exibição dos principais poderes de cada cluster

Perfil físico médio (altura, peso) dos grupos

Lista dos heróis pertencentes a cada cluster

⚖️ 3. Classificação do Alinhamento – Questão 3

Modelo implementado:

Bernoulli Naive Bayes

Funcionalidades:

Selecionar um herói e prever se ele é good ou bad

Comparação com o alinhamento real

Probabilidades da previsão

Exibição dos poderes principais do herói selecionado

⚖️ 4. Classificação alternativa – Questão 4

Além do Naive Bayes, o projeto inclui análise e justificativa do uso do Random Forest Classifier, com comparações técnicas entre:

hipóteses dos modelos

desempenho

robustez

interpretação

(Implementado na análise do case)

⚖️ 5. Regressão – Previsão de Peso – Questão 5

Modelo implementado:

Random Forest Regressor

Funcionalidades:

Predição do peso de um super-herói baseada em:

poderes

altura

Exibição das métricas:

MAE

RMSE

R²

Importância das variáveis

Comparação entre peso real e previsto

▶️ Como rodar a aplicação
1. Instale as dependências
pip install streamlit pandas numpy scikit-learn

2. Coloque os arquivos CSV na mesma pasta do app.py

heroes_information.csv

super_hero_powers.csv

3. Execute o Streamlit
streamlit run app.py

🧠 Insights Técnicos Relevantes

O dataset apresenta valores faltantes, alta dimensionalidade e poderes altamente correlacionados.

Para clustering, PCA foi essencial para reduzir variância e estabilizar os grupos.

Naive Bayes funciona bem com alta dimensionalidade, mas sofre com correlação entre poderes.

Random Forest mostrou melhor desempenho tanto em classificação quanto em regressão, por lidar melhor com não linearidades e interações.

O peso dos heróis é uma variável de alta variância e exige modelos robustos.

📌 Observações finais

Este projeto demonstra:

conhecimento de modelagem supervisionada e não supervisionada

capacidade de integração de modelos em uma aplicação interativa

bom uso de pré-processamento, tratamento de dados e explicação técnica

entrega organizada e orientada ao produto

🙋‍♀️ Autora

Elaine (Lane) Andrade
Cientista de Dados – Desafio Técnico Alelo
Contato disponível no GitHub
