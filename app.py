# Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from io import BytesIO
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="ADII Digital Transformation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stTabs [data-baseweb="tab"] {
        font-size: 1.2rem;
        font-weight: 600;
    }
    h1 { font-size: 2.5rem; }
    h2 { font-size: 2rem; }
    h3 { font-size: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# Utility functions
@st.cache_data
def load_and_validate_data(path: str = 'Donn_es_simul_es_ADII.csv') -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, sep=';')
        df = df.drop_duplicates()
        # Fill numeric NaNs with mean
        df = df.fillna(df.mean(numeric_only=True))
        # Clip numeric to 1-5
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[num_cols] = df[num_cols].clip(1, 5)
        return df
    except FileNotFoundError:
        st.error("Le fichier de données n'a pas été trouvé. Vérifiez le chemin.")
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
    return None

@st.cache_data
def calculate_mean_score(df: pd.DataFrame, cols: list[str]) -> float:
    return df[cols].mean(axis=1).mean()

@st.cache_data
def perform_advanced_analysis(df: pd.DataFrame, x: str, y: str) -> dict:
    # Compute Pearson correlation and p-value
    corr, p = stats.pearsonr(df[x], df[y])
    lin = stats.linregress(df[x], df[y])
    return {'correlation': corr, 'p_value': p, 't_stat': lin.slope / lin.stderr if lin.stderr else np.nan}

@st.cache_data
def generate_insights(results: dict, x: str, y: str) -> list[str]:
    insights = []
    r = results['correlation']
    if abs(r) > 0.7:
        insights.append('Relation forte détectée')
    elif abs(r) > 0.3:
        insights.append('Relation modérée détectée')
    else:
        insights.append('Relation faible ou nulle détectée')
    return insights

@st.cache_data
def export_results(df: pd.DataFrame, scores: dict) -> None:
    # Export summary scores as Excel
    data = pd.DataFrame(list(scores.items()), columns=['Dimension', 'Score'])
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        data.to_excel(writer, sheet_name='Scores', index=False)
        writer.save()
    st.download_button('Télécharger les scores', data=buffer.getvalue(), file_name='scores.xlsx')

# Load data
with st.spinner('Chargement des données...'):
    df = load_and_validate_data()

if df is None or df.empty:
    st.stop()

# Sidebar
st.sidebar.title('Navigation')
st.sidebar.markdown(f"**Observations:** {len(df)}  
**Variables:** {len(df.columns)}")
st.sidebar.markdown(f"**Date:** {df['Date'].min() if 'Date' in df else 'N/A'} - {df['Date'].max() if 'Date' in df else 'N/A'}")

# Define variable dictionary
var_dict = {
    'ADT': {'full_name': 'Adoption digitale', 'items': {f'ADT_{i}': '' for i in range(1,5)}},
    'INT': {'full_name': 'Intention',      'items': {f'INT_{i}': '' for i in range(1,5)}},
    'SAT': {'full_name': 'Satisfaction',   'items': {f'SAT_{i}': '' for i in range(1,5)}},
    'FOR': {'full_name': 'Formation',      'items': {f'FOR_{i}': '' for i in range(1,5)}},
    'RE':  {'full_name': 'Résistance',     'items': {f'RE_{i}': ''  for i in range(1,5)}},
    'PE':  {'full_name': 'Efficacité',      'items': {f'PE_{i}': ''  for i in range(1,5)}},
    'EE':  {'full_name': 'Effort',         'items': {f'EE_{i}': ''  for i in range(1,5)}},
    'FC':  {'full_name': 'Facilitants',    'items': {f'FC_{i}': ''  for i in range(1,5)}},
    'SI':  {'full_name': 'Social',         'items': {f'SI_{i}': ''  for i in range(1,5)}},
}

# Tabs
tabs = st.tabs([
    'Accueil','Univariée','Bivariée','Corrélations','Préparation','Régression','Synthèse'
])

# Tab: Accueil
with tabs[0]:
    st.title('Tableau de Bord ADII')
    st.write('Analyse de la transformation digitale à l'ADII')

# Tab: Univariée
with tabs[1]:
    st.header('Analyse Univariée')
    for key, info in var_dict.items():
        cols = list(info['items'].keys())
        if set(cols).issubset(df.columns):
            mean_score = calculate_mean_score(df, cols)
            st.metric(info['full_name'], f"{mean_score:.2f}/5")
            fig = px.histogram(df, x=df[cols].mean(axis=1), nbins=20,
                               title=f'Distribution {info["full_name"]}')
            st.plotly_chart(fig)

# Tab: Bivariée
with tabs[2]:
    st.header('Analyse Bivariée')
    all_items = [v for info in var_dict.values() for v in info['items'].keys()]
    x = st.selectbox('Variable X', all_items)
    y = st.selectbox('Variable Y', all_items, index=1)
    if x in df.columns and y in df.columns:
        fig = px.scatter(df, x=x, y=y, trendline='ols')
        st.plotly_chart(fig)
        res = perform_advanced_analysis(df, x, y)
        st.json(res)
        for ins in generate_insights(res, x, y): st.write('-', ins)

# Tab: Corrélations
with tabs[3]:
    st.header('Corrélations')
    dims = st.multiselect('Dimensions', list(var_dict.keys()), default=list(var_dict.keys())[:3])
    cols = [f"{d}_{i}" for d in dims for i in range(1,5) if f"{d}_{i}" in df.columns]
    corr = df[cols].corr()
    fig = px.imshow(corr, title='Matrice de Corrélation')
    st.plotly_chart(fig)

# Tab: Préparation
with tabs[4]:
    st.header('Préparation')
    enc = {}
    for col in ['Profil','Sexe','Diplome']:
        if col in df.columns:
            le = LabelEncoder()
            df[col+'_enc'] = le.fit_transform(df[col].fillna(''))
            enc[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    st.json(enc)

# Tab: Régression
with tabs[5]:
    st.header('Régression Linéaire')
    target = st.selectbox('Cible', all_items)
    features = st.multiselect('Features', [c for c in df.columns if c.endswith('_enc')])
    if st.button('Exécuter') and target in df.columns and features:
        X = df[features]; y = df[target]
        model = LinearRegression().fit(X, y)
        pred = model.predict(X)
        st.metric('R²', f"{r2_score(y, pred):.3f}")
        coefs = pd.DataFrame({'Feature': features, 'Coef': model.coef_})
        st.bar_chart(coefs.set_index('Feature'))

# Tab: Synthèse
with tabs[6]:
    st.header('Synthèse')
    scores = {info['full_name']: calculate_mean_score(df, list(info['items'].keys())) for info in var_dict.values()}
    categories, values = list(scores.keys()), list(scores.values())
    fig = go.Figure(go.Scatterpolar(r=values+[values[0]], theta=categories+[categories[0]], fill='toself'))
    st.plotly_chart(fig)
    export_results(df, scores)
