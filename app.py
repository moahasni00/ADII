# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score, classification_report
from io import BytesIO
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="ADII Digital Transformation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Utility Functions ---
@st.cache_data
def load_and_validate_data(path: str = 'Donn_es_simul_es_ADII.csv') -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, sep=';')
        df = df.drop_duplicates()
        df = df.fillna(df.mean(numeric_only=True))
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

# --- Load Data ---
with st.spinner('Chargement des données...'):
    df = load_and_validate_data()

if df is None or df.empty:
    st.stop()

# --- Sidebar Summary ---
st.sidebar.title('💡 Résumé des Données')
st.sidebar.markdown(f"**Observations:** {len(df)}  ")
st.sidebar.markdown(f"**Variables:** {len(df.columns)}")
st.sidebar.markdown(f"**Date:** {df['Date'].min() if 'Date' in df else 'N/A'} - {df['Date'].max() if 'Date' in df else 'N/A'}")

# --- Variable Dictionary ---
var_dict = {
    'ADT': {'full_name': 'Adoption Digitale',    'items': [f'ADT_{i}' for i in range(1,5)]},
    'INT': {'full_name': 'Intention d’Usage',   'items': [f'INT_{i}' for i in range(1,5)]},
    'SAT': {'full_name': 'Satisfaction',        'items': [f'SAT_{i}' for i in range(1,5)]},
    'FOR': {'full_name': 'Formation & Support', 'items': [f'FOR_{i}' for i in range(1,5)]},
    'RE':  {'full_name': 'Résistance au Changement', 'items': [f'RE_{i}' for i in range(1,5)]},
    'PE':  {'full_name': 'Performance Attendue','items': [f'PE_{i}' for i in range(1,5)]},
    'EE':  {'full_name': 'Effort Attendu',     'items': [f'EE_{i}' for i in range(1,5)]},
    'FC':  {'full_name': 'Conditions Facilitantes','items': [f'FC_{i}' for i in range(1,5)]},
    'SI':  {'full_name': 'Influence Sociale',   'items': [f'SI_{i}' for i in range(1,5)]}
}

# --- Tabs ---
tabs = st.tabs([
    'Accueil',
    'Analyse Univariée',
    'Analyse Bivariée',
    'Corrélations',
    'Préparation',
    'Modélisation',
    'Synthèse'
])

# --- Tab: Accueil ---
with tabs[0]:
    st.title('📊 Tableau de Bord – Transformation Digitale ADII')
    st.markdown(
        "Cette application analyse les résultats de l'enquête ADII sur la transformation digitale,\n"
        "avec visualisations interactives et analyses statistiques."
    )

# --- Tab: Analyse Univariée ---
with tabs[1]:
    st.header('📊 Analyse Univariée & Réponses')
    for key, info in var_dict.items():
        cols = info['items']
        if not set(cols).issubset(df.columns):
            continue
        st.subheader(info['full_name'])
        # Raw responses sample
        st.markdown('**Exemple de réponses brutes :**')
        st.dataframe(df[cols].head(5))
        # Descriptive statistics
        desc = df[cols].describe().T[['count','mean','std','min','25%','50%','75%','max']]
        st.markdown('**Statistiques descriptives :**')
        st.dataframe(desc)
        # Distribution per question
        for col in cols:
            counts = df[col].value_counts().reindex([1,2,3,4,5], fill_value=0)
            fig = px.bar(
                x=counts.index, y=counts.values,
                labels={'x':'Échelle (1-5)','y':'Nombre de réponses'},
                title=f'{col} : Distribution des réponses'
            )
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('---')

# --- Tab: Analyse Bivariée ---
with tabs[2]:
    st.header('🔄 Analyse Bivariée')
    all_items = [item for info in var_dict.values() for item in info['items']]
    x = st.selectbox('Variable X', all_items)
    y = st.selectbox('Variable Y', all_items, index=1)
    if x in df.columns and y in df.columns:
        fig = px.scatter(df, x=x, y=y, trendline='ols',
                         title=f'Relation entre {x} et {y}')
        st.plotly_chart(fig)

# --- Tab: Corrélations ---
with tabs[3]:
    st.header('🔗 Corrélations')
    dims = st.multiselect('Dimensions', list(var_dict.keys()), default=list(var_dict.keys())[:3])
    cols = [f"{d}_{i}" for d in dims for i in range(1,5) if f"{d}_{i}" in df.columns]
    corr = df[cols].corr()
    fig = px.imshow(corr, title='Matrice de Corrélation')
    st.plotly_chart(fig, use_container_width=True)

# --- Tab: Préparation ---
with tabs[4]:
    st.header('🛠️ Préparation des Données')
    encoders = {}
    for col in ['Profil','Sexe','Diplome']:
        if col in df.columns:
            le = LabelEncoder()
            df[col+'_enc'] = le.fit_transform(df[col].fillna(''))
            encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    st.json(encoders)

# --- Tab: Modélisation ---
with tabs[5]:
    st.header('📈 Modélisation & Analyses Statistiques')
    model_type = st.selectbox(
        "Type de modèle",
        ['Régression Linéaire', 'Régression Logistique']
    )
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target = st.selectbox("Variable cible", numeric_cols)
    features = st.multiselect(
        "Variables explicatives",
        [c for c in numeric_cols if c != target],
        default=numeric_cols[:3]
    )

    if st.button("Entraîner le modèle"):
        if not features:
            st.error("Sélectionnez au moins une variable explicative.")
        else:
            X = df[features]
            y = df[target]
            if model_type == 'Régression Linéaire':
                model = LinearRegression()
                model.fit(X, y)
                preds = model.predict(X)
                st.metric("R² Score", f"{r2_score(y, preds):.3f}")
                coef_df = pd.DataFrame({'Variable': features, 'Coefficient': model.coef_}).set_index('Variable')
                st.bar_chart(coef_df)
            else:
                if y.nunique() != 2:
                    st.error("⚠️ La variable cible doit être binaire pour la régression logistique.")
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.3, random_stat
