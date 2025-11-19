
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ---------------------------------------------------------
# CONFIGURAÇÃO INICIAL DA PÁGINA
# ---------------------------------------------------------
st.set_page_config(
    page_title="Super-Heróis – Exploração e Modelos",
    layout="wide"
)

st.title("🦸‍♀️ Super-Heróis: Exploração e Modelos de Machine Learning")
st.markdown(
    """
Aplicação interativa para explorar os dados de super-heróis e interagir com:

- **Clustering** dos heróis pelos poderes (Questão 1)  
- **Classificação** de alinhamento (good/bad) usando Naive Bayes (Questão 3)  
- **Regressão** para prever peso (Questão 5)  

Use o menu lateral para navegar entre as seções.
"""
)

# ---------------------------------------------------------
# CARREGAMENTO E PRÉ-PROCESSAMENTO BÁSICO
# ---------------------------------------------------------

@st.cache_data
def load_data():
    # Substitua pelos caminhos corretos se necessário
    info = pd.read_csv("heroes_information.csv")
    powers = pd.read_csv("super_hero_powers.csv")

    # Ajustes básicos
    info = info.replace(-99, np.nan)
    info = info.rename(columns={"name": "hero_names"})

    # Merge
    df = pd.merge(info, powers, on="hero_names", how="inner")

    # Lista de colunas de poderes (todas exceto hero_names)
    power_cols = list(powers.columns[1:])

    # Garantir que poderes sejam 0/1
    df[power_cols] = df[power_cols].fillna(False).astype(int)

    return info, powers, df, power_cols


info, powers, df, power_cols = load_data()

# ---------------------------------------------------------
# FUNÇÕES DE MODELAGEM (COM CACHE)
# ---------------------------------------------------------

@st.cache_resource
def train_clustering(k_clusters: int = 4):
    """
    Treina PCA + KMeans para clustering dos heróis
    usando poderes + altura + peso.
    """
    df_clust = df.copy()

    # Features: poderes + Height + Weight
    features = power_cols + ["Height", "Weight"]
    X = df_clust[features].copy()

    # Tratar NaN em Height e Weight
    X["Height"] = X["Height"].fillna(X["Height"].median())
    X["Weight"] = X["Weight"].fillna(X["Weight"].median())

    # Padronização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA para reduzir dimensionalidade (95% da variância)
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # KMeans
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_pca)

    df_clust["cluster"] = clusters

    return df_clust, X_pca, pca, kmeans


@st.cache_resource
def train_naive_bayes():
    """
    Treina Bernoulli Naive Bayes para prever Alignment (good/bad)
    usando apenas os poderes (0/1).
    """
    df_nb = df.copy()
    df_nb = df_nb[df_nb["Alignment"].isin(["good", "bad"])].copy()
    df_nb["target"] = df_nb["Alignment"].map({"good": 1, "bad": 0})

    X = df_nb[power_cols].copy()
    X = X.fillna(0).astype(int)
    y = df_nb["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    model = BernoulliNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, df_nb, X_train, X_test, y_train, y_test, acc


@st.cache_resource
def train_regressor():
    """
    Treina Random Forest Regressor para prever Weight
    usando poderes + Height.
    """
    df_reg = df.copy()
    df_reg = df_reg[df_reg["Weight"].notna()].copy()
    df_reg = df_reg[df_reg["Weight"] > 0].copy()

    features = power_cols + ["Height"]

    X = df_reg[features].copy()
    y = df_reg["Weight"].copy()

    # Tratar NaN
    X[power_cols] = X[power_cols].fillna(0)
    X["Height"] = X["Height"].fillna(X["Height"].median())

    X = X.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42
    )

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

    return rf, df_reg, X_train, X_test, y_train, y_test, mae, rmse, r2, importances


# ---------------------------------------------------------
# MENU LATERAL
# ---------------------------------------------------------
menu = st.sidebar.radio(
    "📌 Navegação",
    [
        "Documentação",
        "Exploração de Dados",
        "Clustering (Grupos de Heróis)",
        "Classificação (Alinhamento)",
        "Regressão (Peso)"
    ]
)

# ---------------------------------------------------------
# 1. DOCUMENTAÇÃO
# ---------------------------------------------------------
if menu == "Documentação":
    st.header("📖 Documentação e Instruções de Uso")

    st.markdown(
        """
### Visão geral

Esta aplicação foi desenvolvida para:

- Explorar os dados dos super-heróis;
- Visualizar agrupamentos (clustering) de heróis com base em seus poderes;
- Classificar o alinhamento (good/bad);
- Prever o peso de um herói a partir de suas características.

### Como usar

- **Exploração de Dados**  
  Veja as tabelas, estatísticas descritivas, distribuições e aplique filtros por alinhamento, gênero e editora.

- **Clustering (Grupos de Heróis)**  
  Visualize os clusters formados a partir dos poderes e características físicas.  
  Selecione um cluster para ver os principais poderes e o perfil médio do grupo.

- **Classificação (Alinhamento)**  
  Selecione um herói e veja a previsão de alinhamento (good/bad) pelo modelo Naive Bayes, 
  além da comparação com o alinhamento real.

- **Regressão (Peso)**  
  Selecione um herói e veja o peso previsto pelo modelo de regressão Random Forest 
  em comparação com o peso real (quando disponível).

Todos os modelos são treinados automaticamente a partir dos arquivos:
- `heroes_information.csv`
- `super_hero_powers.csv`
"""
    )

# ---------------------------------------------------------
# 2. EXPLORAÇÃO DE DADOS
# ---------------------------------------------------------
elif menu == "Exploração de Dados":
    st.header("🔍 Exploração de Dados")

    st.subheader("Filtros básicos")
    col1, col2, col3 = st.columns(3)

    with col1:
        align_filter = st.selectbox(
            "Filtrar por Alignment:",
            options=["Todos"] + sorted(info["Alignment"].dropna().unique().tolist())
        )
    with col2:
        gender_filter = st.selectbox(
            "Filtrar por Gender:",
            options=["Todos"] + sorted(info["Gender"].dropna().unique().tolist())
        )
    with col3:
        publisher_filter = st.selectbox(
            "Filtrar por Publisher:",
            options=["Todos"] + sorted(info["Publisher"].dropna().unique().tolist())
        )

    df_view = info.copy()

    if align_filter != "Todos":
        df_view = df_view[df_view["Alignment"] == align_filter]
    if gender_filter != "Todos":
        df_view = df_view[df_view["Gender"] == gender_filter]
    if publisher_filter != "Todos":
        df_view = df_view[df_view["Publisher"] == publisher_filter]

    st.markdown("#### Tabela de heróis filtrada")
    st.dataframe(df_view)

    st.markdown("#### Estatísticas descritivas (numéricas)")
    st.write(df_view.describe())

    st.markdown("#### Distribuição de Alignment")
    st.bar_chart(df_view["Alignment"].value_counts())

    st.markdown("#### Distribuição de Height (ignora valores faltantes)")
    st.bar_chart(df_view["Height"].dropna())

# ---------------------------------------------------------
# 3. CLUSTERING
# ---------------------------------------------------------
elif menu == "Clustering (Grupos de Heróis)":
    st.header("🧩 Clustering – Grupos de Heróis")

    st.markdown(
        """
Os heróis foram agrupados com base em seus poderes e características físicas
(utilizando PCA + KMeans).  
Use o seletor abaixo para definir o número de clusters.
"""
    )

    k = st.slider("Número de clusters (K)", min_value=2, max_value=8, value=4, step=1)
    df_clust, X_pca, pca_model, kmeans_model = train_clustering(k)

    st.markdown("#### Visualização em 2 componentes principais (PCA)")
    # Usar apenas as duas primeiras componentes para o gráfico
    plot_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "cluster": df_clust["cluster"].astype(str),
        "hero_names": df_clust["hero_names"]
    })

    st.scatter_chart(
        plot_df,
        x="PC1",
        y="PC2",
        color="cluster"
    )

    st.markdown("#### Perfil dos clusters")

    cluster_ids = sorted(df_clust["cluster"].unique().tolist())
    selected_cluster = st.selectbox(
        "Selecione um cluster para explorar:",
        options=cluster_ids
    )

    cluster_data = df_clust[df_clust["cluster"] == selected_cluster]

    st.write(f"Número de heróis no cluster {selected_cluster}: **{len(cluster_data)}**")

    st.write("Altura mediana:", cluster_data["Height"].median())
    st.write("Peso mediano:", cluster_data["Weight"].median())

    mean_powers = cluster_data[power_cols].mean().sort_values(ascending=False)
    top_powers = mean_powers.head(10)

    st.markdown("Principais poderes (frequência média dentro do cluster):")
    st.table(top_powers.to_frame("Frequência"))

    st.markdown("Alguns heróis deste cluster:")
    st.write(cluster_data["hero_names"].head(20).tolist())

# ---------------------------------------------------------
# 4. CLASSIFICAÇÃO (NAIVE BAYES)
# ---------------------------------------------------------
elif menu == "Classificação (Alinhamento)":
    st.header("⚖️ Classificação – Alinhamento (good/bad)")

    model_nb, df_nb, X_train_nb, X_test_nb, y_train_nb, y_test_nb, acc_nb = train_naive_bayes()

    st.write(f"Acurácia do Naive Bayes (teste): **{acc_nb:.3f}**")

    st.markdown(
        """
Selecione um herói com alinhamento conhecido (`good` ou `bad`) 
para ver a previsão do modelo e comparar com o valor real.
"""
    )

    hero_options = df_nb["hero_names"].sort_values().unique().tolist()
    selected_hero = st.selectbox("Escolha um herói:", hero_options)

    hero_row = df_nb[df_nb["hero_names"] == selected_hero].iloc[0]

    X_hero = hero_row[power_cols].values.reshape(1, -1)
    pred = model_nb.predict(X_hero)[0]
    proba = model_nb.predict_proba(X_hero)[0]

    pred_label = "good" if pred == 1 else "bad"
    real_label = hero_row["Alignment"]

    st.write(f"**Alinhamento real:** {real_label}")
    st.write(f"**Previsão do modelo:** {pred_label}")
    st.write(f"Probabilidades (Naive Bayes): good = {proba[1]:.3f}, bad = {proba[0]:.3f}")

    st.markdown("Poderes principais deste herói (valor = 1):")
    hero_powers_true = hero_row[power_cols][hero_row[power_cols] == 1].index.tolist()
    st.write(hero_powers_true if hero_powers_true else "Nenhum poder marcado como 1.")

# ---------------------------------------------------------
# 5. REGRESSÃO (PESO)
# ---------------------------------------------------------
elif menu == "Regressão (Peso)":
    st.header("⚖️ Regulação – Previsão de Peso")

    rf_reg, df_reg, X_train_reg, X_test_reg, y_train_reg, y_test_reg, mae, rmse, r2, importances = train_regressor()

    st.markdown("#### Métricas de desempenho do modelo (Random Forest Regressor)")
    st.write(f"MAE (erro absoluto médio): **{mae:.2f}**")
    st.write(f"RMSE (raiz do erro quadrático médio): **{rmse:.2f}**")
    st.write(f"R² (coeficiente de determinação): **{r2:.3f}**")

    st.markdown("#### Principais variáveis para prever o peso")
    st.table(importances.head(10).to_frame("Importância"))

    st.markdown(
        """
Selecione um herói com peso conhecido para ver a previsão do modelo
e comparar com o valor real.
"""
    )

    hero_options_reg = df_reg["hero_names"].sort_values().unique().tolist()
    selected_hero_reg = st.selectbox("Escolha um herói:", hero_options_reg)

    hero_row_reg = df_reg[df_reg["hero_names"] == selected_hero_reg].iloc[0]

    # Montar vetor de features
    features = power_cols + ["Height"]
    X_hero_reg = hero_row_reg[features].copy()

    # Tratar NaN para o herói selecionado
    X_hero_reg[power_cols] = X_hero_reg[power_cols].fillna(0)
    X_hero_reg["Height"] = (
        X_hero_reg["Height"]
        if pd.notna(X_hero_reg["Height"])
        else df_reg["Height"].median()
    )

    X_hero_reg = X_hero_reg.values.reshape(1, -1)

    pred_weight = rf_reg.predict(X_hero_reg)[0]
    real_weight = hero_row_reg["Weight"]

    st.write(f"**Peso real:** {real_weight} (quando disponível)")
    st.write(f"**Peso previsto pelo modelo:** {pred_weight:.2f}")

    st.markdown("Poderes principais deste herói (marcados com 1):")
    hero_powers_true_reg = hero_row_reg[power_cols][hero_row_reg[power_cols] == 1].index.tolist()
    st.write(hero_powers_true_reg if hero_powers_true_reg else "Nenhum poder marcado como 1.")
