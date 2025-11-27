"""
Super-Heroes Analytics — Versão Streamlit (layout executivo)
- Uso: streamlit run app.py
- Coloque os arquivos heroes_information.csv e super_hero_powers.csv
  na mesma pasta OU faça upload via interface.
- Inclui: exploração, clustering (KMeans), classificação (Naive Bayes),
  regressão de peso (LinearRegression) e documentação de uso na UI.
"""

from typing import Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ---------------------------
# Config page
# ---------------------------
st.set_page_config(
    page_title="Super-Heroes Analytics — Desafio Alelo",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# Utility: Load Data
# ---------------------------
@st.cache_data(show_spinner=False)
def load_csv_safe(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def load_data_from_files(heroes_path: Optional[str], powers_path: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # If user provided paths, try load; otherwise create empty frames
    heroes_df = pd.DataFrame()
    powers_df = pd.DataFrame()
    if heroes_path:
        heroes_df = load_csv_safe(heroes_path)
    if powers_path:
        powers_df = load_csv_safe(powers_path)
    return heroes_df, powers_df

# ---------------------------
# Sidebar: Inputs & Uploads
# ---------------------------
st.sidebar.title("Controles")
st.sidebar.markdown("Faça upload dos datasets ou deixe em branco para carregar do diretório atual.")

heroes_file = st.sidebar.file_uploader("Upload: heroes_information.csv", type=["csv"])
powers_file = st.sidebar.file_uploader("Upload: super_hero_powers.csv", type=["csv"])

use_local_files = False
if not heroes_file or not powers_file:
    # Tenta carregar arquivos locais automaticamente
    try:
        heroes_local = "heroes_information.csv"
        powers_local = "super_hero_powers.csv"
        # carrega se existirem
        heroes_df_local = load_csv_safe(heroes_local)
        powers_df_local = load_csv_safe(powers_local)
        use_local_files = True
    except Exception:
        heroes_df_local = pd.DataFrame()
        powers_df_local = pd.DataFrame()
else:
    heroes_df_local = pd.DataFrame()
    powers_df_local = pd.DataFrame()

if heroes_file:
    heroes_df = pd.read_csv(heroes_file)
else:
    heroes_df = heroes_df_local

if powers_file:
    powers_df = pd.read_csv(powers_file)
else:
    powers_df = powers_df_local

# If still empty, show instruction and stop early
if heroes_df.empty and powers_df.empty:
    st.title("Super-Heroes Analytics — Desafio Alelo")
    st.warning(
        "Nenhum dataset fornecido. Por favor faça upload dos arquivos CSV no painel lateral "
        "ou coloque 'heroes_information.csv' e 'super_hero_powers.csv' na mesma pasta deste script."
    )
    st.stop()

# ---------------------------
# Preprocess helpers
# ---------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Normaliza nomes de colunas para evitar mismatch
    df = df.rename(columns=lambda x: x.strip())
    return df

heroes_df = normalize_columns(heroes_df)
powers_df = normalize_columns(powers_df)

# Tenta harmonizar colunas comuns
# Algumas versões do dataset usam 'Alignment' vs 'alignment', 'Weight' vs 'Weight lbs', etc.
def harmonize_heroes(df: pd.DataFrame) -> pd.DataFrame:
    # copy to avoid side-effects
    df = df.copy()
    # common fixes
    candidates = {c.lower(): c for c in df.columns}
    # lowercase map
    lowermap = {c.lower(): c for c in df.columns}
    # unify some column names if present
    if 'alignment' in lowermap:
        df = df.rename(columns={lowermap['alignment']: 'Alignment'})
    if 'gender' in lowermap:
        df = df.rename(columns={lowermap['gender']: 'Gender'})
    # Height & Weight normalization (keep numeric part)
    for col in df.columns:
        if col.lower().startswith("height"):
            df = df.rename(columns={col: 'Height'})
        if col.lower().startswith("weight"):
            df = df.rename(columns={col: 'Weight'})
    return df

heroes_df = harmonize_heroes(heroes_df)

# Convert Height/Weight to numeric if possible (extract numeric)
def to_numeric_strip(x):
    try:
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            # remove non-digit, dot, minus
            s = ''.join(ch for ch in x if ch.isdigit() or ch in ".-")
            return float(s) if s != "" else np.nan
        return float(x)
    except Exception:
        return np.nan

if 'Height' in heroes_df.columns:
    heroes_df['Height'] = heroes_df['Height'].apply(to_numeric_strip)

if 'Weight' in heroes_df.columns:
    heroes_df['Weight'] = heroes_df['Weight'].apply(to_numeric_strip)

# ---------------------------
# App Layout: Header
# ---------------------------
st.markdown("<h1 style='margin-bottom:6px'>Super-Heroes Analytics — Desafio Alelo</h1>", unsafe_allow_html=True)
st.markdown("<small>Exploração, clustering, classificação e regressão — interface executiva</small>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------------
# Section: Dataset Explorer
# ---------------------------
st.header("Exploração dos Dados")
col1, col2 = st.columns([1, 2])

with col1:
    ds_choice = st.selectbox("Dataset para visualizar", ["Heroes Information", "Super Hero Powers"])
    nrows = st.number_input("Linhas a mostrar", min_value=3, max_value=100, value=10, step=1)
    show_stats = st.checkbox("Mostrar estatísticas descritivas", value=True)

with col2:
    # quick filters
    st.markdown("**Filtros rápidos (heroes)**")
    if 'Alignment' in heroes_df.columns:
        align_vals = sorted(heroes_df['Alignment'].dropna().unique().tolist())
        selected_align = st.multiselect("Alignment", options=align_vals, default=align_vals)
    else:
        selected_align = []
    if 'Gender' in heroes_df.columns:
        gender_vals = sorted(heroes_df['Gender'].dropna().unique().tolist())
        selected_gender = st.multiselect("Gender", options=gender_vals, default=gender_vals)
    else:
        selected_gender = []
    # publisher / publisherName
    pub_cols = [c for c in heroes_df.columns if 'publisher' in c.lower() or 'publisher' in ' '.join(heroes_df.columns).lower()]
    if pub_cols:
        pub_col = pub_cols[0]
        pub_vals = sorted(heroes_df[pub_col].dropna().unique().tolist())
        selected_publisher = st.multiselect("Publisher", options=pub_vals, default=pub_vals)
    else:
        pub_col = None
        selected_publisher = []

st.write("")  # spacing

# show chosen dataset
if ds_choice == "Heroes Information":
    df_show = heroes_df.copy()
else:
    df_show = powers_df.copy()

# Apply filters (if on heroes dataset)
if ds_choice == "Heroes Information" and not df_show.empty:
    if 'Alignment' in df_show.columns and selected_align:
        df_show = df_show[df_show['Alignment'].isin(selected_align)]
    if 'Gender' in df_show.columns and selected_gender:
        df_show = df_show[df_show['Gender'].isin(selected_gender)]
    if pub_col and selected_publisher:
        df_show = df_show[df_show[pub_col].isin(selected_publisher)]

# Display dataframe and stats
st.subheader("Visualização")
st.dataframe(df_show.head(nrows), use_container_width=True)

if show_stats and not df_show.empty:
    st.subheader("Estatísticas descritivas (numéricas)")
    st.dataframe(df_show.describe(include=[np.number]).T, use_container_width=True)

st.markdown("---")

# ---------------------------
# Section: Clustering (KMeans)
# ---------------------------
st.header("Análise de Clusters (KMeans)")

# Prepare data for clustering: use Height/Weight and optionally other numeric features
cluster_cols = []
if 'Height' in heroes_df.columns and 'Weight' in heroes_df.columns:
    cluster_cols = ['Height', 'Weight']
else:
    # find numeric columns
    cluster_cols = heroes_df.select_dtypes(include=[np.number]).columns.tolist()

if len(cluster_cols) < 2:
    st.warning("Não há colunas numéricas suficientes para gerar um clustering significativo (precisa de pelo menos 2).")
else:
    n_clusters = st.slider("Número de clusters (K)", 2, 8, 3)
    run_cluster = st.button("Gerar Clustering")
    if run_cluster:
        # prepare data
        cluster_df = heroes_df[cluster_cols].dropna()
        X = cluster_df.values
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        preds = kmeans.fit_predict(Xs)
        cluster_df = cluster_df.assign(cluster=preds)

        # Project to 2D for plotting: if more than 2 dimensions, use the first two PC-like axes (here we'll use scaled first two cols)
        x_axis = cluster_cols[0]
        y_axis = cluster_cols[1] if len(cluster_cols) > 1 else cluster_cols[0]

        # Merge cluster labels back into a sample of hero names if available
        result_df = heroes_df.loc[cluster_df.index].copy()
        result_df = result_df.reset_index(drop=True)
        cluster_df = cluster_df.reset_index(drop=True)
        result_df['cluster'] = cluster_df['cluster']

        # Interactive scatter with Plotly
        fig = px.scatter(
            result_df,
            x=x_axis, y=y_axis,
            color=result_df['cluster'].astype(str),
            hover_data=result_df.columns,
            title=f"KMeans (k={n_clusters}) — {x_axis} x {y_axis}",
            width=900, height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show cluster summary
        st.subheader("Resumo por cluster")
        summary = result_df.groupby('cluster')[[x_axis, y_axis]].agg(['count', 'mean', 'median']).round(2)
        st.dataframe(summary, use_container_width=True)

        # Interpretation text
        st.markdown("**Interpretação rápida:**")
        st.markdown(
            "Os clusters acima agrupam heróis com características semelhantes em altura e peso. "
            "Use o resumo por cluster para identificar grupos de 'mais altos/pesados', 'baixos/leves' etc. "
            "Para aprofundar, experimente alterar o número de clusters (K)."
        )

st.markdown("---")

# ---------------------------
# Section: Classification (Naive Bayes)
# ---------------------------
st.header("Classificação de Alinhamento (Naive Bayes)")

# Prepare classifier if possible
can_train_clf = all(col in heroes_df.columns for col in ['Alignment', 'Height', 'Weight', 'Gender'])
if not can_train_clf:
    st.info("Dados insuficientes para treinar o classificador. Os dados precisam conter: Alignment, Height, Weight, Gender.")
else:
    with st.expander("Treinar modelo Naive Bayes (ver detalhes)"):
        st.write("O classificador usa Height, Weight e Gender como features.")
        if st.button("Treinar modelo agora"):
            df_clf = heroes_df[['Alignment', 'Height', 'Weight', 'Gender']].dropna().copy()
            # Encode gender (categorical)
            df_clf['Gender_code'] = pd.Categorical(df_clf['Gender']).codes
            X = df_clf[['Height', 'Weight', 'Gender_code']]
            y = df_clf['Alignment']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = GaussianNB()
            clf.fit(X_train, y_train)
            acc = clf.score(X_test, y_test)
            st.success(f"Modelo treinado. Acurácia (test): {acc:.3f}")
            # cache model in session state
            st.session_state['alignment_model'] = clf

    st.markdown("**Predizer alinhamento para um herói**")
    c1, c2, c3 = st.columns(3)
    with c1:
        in_height = st.number_input("Height (para predição)", value=float(180.0))
    with c2:
        in_weight = st.number_input("Weight (para predição)", value=float(80.0))
    with c3:
        in_gender = st.selectbox("Gender", options=sorted(heroes_df['Gender'].dropna().unique().tolist()))

    if st.button("Prever Alinhamento"):
        if 'alignment_model' not in st.session_state:
            # train on the fly
            df_clf = heroes_df[['Alignment', 'Height', 'Weight', 'Gender']].dropna().copy()
            df_clf['Gender_code'] = pd.Categorical(df_clf['Gender']).codes
            X = df_clf[['Height', 'Weight', 'Gender_code']]
            y = df_clf['Alignment']
            clf = GaussianNB()
            clf.fit(X, y)
            st.session_state['alignment_model'] = clf
            # Also keep mapping
            st.session_state['gender_categories'] = list(pd.Categorical(df_clf['Gender']).categories)
        model = st.session_state['alignment_model']
        # map gender to code (attempt find)
        try:
            gender_categories = st.session_state.get('gender_categories', list(pd.Categorical(heroes_df['Gender']).categories))
            gender_code = gender_categories.index(in_gender)
        except Exception:
            gender_code = 0
        pred = model.predict([[in_height, in_weight, gender_code]])[0]
        proba = model.predict_proba([[in_height, in_weight, gender_code]]) if hasattr(model, "predict_proba") else None
        st.write(f"**Previsão:** {pred}")
        if proba is not None:
            # show class probabilities
            classes = model.classes_
            probs = dict(zip(classes, proba[0].round(3)))
            st.write("Probabilidades:", probs)

st.markdown("---")

# ---------------------------
# Section: Regression (Weight prediction)
# ---------------------------
st.header("Previsão de Peso (Regressão Linear simples)")

can_train_reg = 'Height' in heroes_df.columns and 'Weight' in heroes_df.columns
if not can_train_reg:
    st.info("Dados insuficientes para treinar regressão (necessita Height e Weight).")
else:
    with st.expander("Treinar / visualizar regressão"):
        df_reg = heroes_df[['Height', 'Weight']].dropna()
        st.write("Ajuste por regressão linear simples (Weight ~ Height).")
        if st.button("Treinar regressão"):
            X = df_reg[['Height']].values.reshape(-1, 1)
            y = df_reg['Weight'].values
            reg = LinearRegression()
            reg.fit(X, y)
            st.session_state['reg_model'] = reg
            st.success("Regressão treinada e armazenada na sessão.")
            st.write(f"Coeficiente (slope): {reg.coef_[0]:.4f} ; Intercept: {reg.intercept_:.2f}")

        # plot scatter + regression line if model exists or on-the-fly
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.scatterplot(data=df_reg, x='Height', y='Weight', ax=ax, alpha=0.6)
        if 'reg_model' in st.session_state:
            reg = st.session_state['reg_model']
            xs = np.linspace(df_reg['Height'].min(), df_reg['Height'].max(), 100)
            ys = reg.predict(xs.reshape(-1, 1))
            ax.plot(xs, ys, linewidth=2, label='Regressão')
            ax.legend()
        ax.set_title("Height x Weight (scatter + regressão)")
        st.pyplot(fig)

    st.markdown("**Prever peso a partir da altura**")
    pred_height = st.number_input("Height para prever peso", value=175.0)
    if st.button("Prever Peso"):
        if 'reg_model' not in st.session_state:
            # train quickly if not present
            X = df_reg[['Height']].values.reshape(-1, 1)
            y = df_reg['Weight'].values
            reg = LinearRegression()
            reg.fit(X, y)
            st.session_state['reg_model'] = reg
        reg = st.session_state['reg_model']
        pred_weight = reg.predict(np.array([[pred_height]]))[0]
        st.success(f"Peso previsto: {pred_weight:.2f} (unidades conforme dataset)")

st.markdown("---")

# ---------------------------
# Section: Documentation / Instruções
# ---------------------------
st.header("Documentação e instruções rápidas")
st.markdown(
    """
    **Como usar este app**  
    1. Faça upload dos CSVs (ou coloque na mesma pasta que este script).  
    2. Explore as tabelas na seção "Exploração dos Dados". Use filtros para reduzir o subconjunto.  
    3. Em "Análise de Clusters", ajuste K e gere os clusters para inspecionar grupos.  
    4. Em "Classificação", treine o modelo e use os controles para prever alinhamento.  
    5. Em "Previsão de Peso", treine a regressão e faça previsões a partir da altura.  

    **Observações técnicas**  
    - O pré-processamento tenta extrair números de colunas Height/Weight textuais.  
    - Modelos são simples (Naive Bayes e Regressão Linear) e treinados localmente; para produção recomenda-se validação e features adicionais.  
    """
)

# Footer
st.markdown("---")
st.markdown("**Desenvolvido para o desafio técnico Alelo — versão local.**")
st.caption("Para entrega: crie um repositório GitHub com este script, os CSVs e um README explicativo.")



