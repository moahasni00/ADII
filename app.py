# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score, classification_report
from io import BytesIO

# --- Page Configuration ---
st.set_page_config(
    page_title="ADII Digital Transformation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Data Loading & Validation ---
@st.cache_data
def load_data(path: str = 'Donn_es_simul_es_ADII.csv') -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, sep=';')
        df = df.drop_duplicates()
        df = df.fillna(df.mean(numeric_only=True))
        num_cols = df.select_dtypes(include=['float64','int64']).columns
        df[num_cols] = df[num_cols].clip(1, 5)
        return df
    except FileNotFoundError:
        st.error("Le fichier de données n'a pas été trouvé. Vérifiez le chemin.")
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
    return None

# Load dataset
df = load_data()
if df is None or df.empty:
    st.stop()

# --- Sidebar Summary ---
st.sidebar.title("💡 Résumé des Données")
st.sidebar.write(f"Observations: {len(df)}")
st.sidebar.write(f"Variables: {len(df.columns)}")
if 'Date' in df.columns:
    st.sidebar.write(f"Période: {df['Date'].min()} à {df['Date'].max()}")

# --- Variable Dictionary ---
var_dict = {
    'ADT': {'name': 'Adoption Digitale',      'cols': [f'ADT_{i}' for i in range(1,5)]},
    'INT': {'name': 'Intention d’Usage',     'cols': [f'INT_{i}' for i in range(1,5)]},
    'SAT': {'name': 'Satisfaction',          'cols': [f'SAT_{i}' for i in range(1,5)]},
    'FOR': {'name': 'Formation & Support',   'cols': [f'FOR_{i}' for i in range(1,5)]},
    'RE':  {'name': 'Résistance au Changement','cols': [f'RE_{i}' for i in range(1,5)]},
    'PE':  {'name': 'Performance Attendue',  'cols': [f'PE_{i}' for i in range(1,5)]},
    'EE':  {'name': 'Effort Attendu',       'cols': [f'EE_{i}' for i in range(1,5)]},
    'FC':  {'name': 'Conditions Facilitantes','cols': [f'FC_{i}' for i in range(1,5)]},
    'SI':  {'name': 'Influence Sociale',     'cols': [f'SI_{i}' for i in range(1,5)]}
}

# --- Tabs Creation ---
tabs = st.tabs([
    'Accueil',
    'Univariée',
    'Bivariée',
    'Corrélations',
    'Préparation',
    'Modélisation',
    'Synthèse'
])

# --- Tab: Accueil ---
with tabs[0]:
    st.title("📊 Tableau de Bord – Transformation Digitale ADII")
    st.markdown(
        "Cette application analyse les résultats de l'enquête ADII sur la transformation digitale, "
        "avec visualisations interactives et analyses statistiques."
    )

# --- Tab: Univariée ---
with tabs[1]:
    st.header("📊 Analyse Univariée & Réponses")
    for key, info in var_dict.items():
        cols = info['cols']
        if set(cols).issubset(df.columns):
            st.subheader(info['name'])
            # Sample responses
            st.markdown("**Exemple de réponses brutes:**")
            st.dataframe(df[cols].head(5))
            # Descriptive stats
            desc = df[cols].describe().T[['count','mean','std','min','25%','50%','75%','max']]
            st.markdown("**Statistiques descriptives:**")
            st.dataframe(desc)
            # Distribution per question
            for col in cols:
                counts = df[col].value_counts().reindex([1,2,3,4,5], fill_value=0)
                fig = px.bar(
                    x=counts.index,
                    y=counts.values,
                    labels={'x':'Échelle','y':'Nombre'},
                    title=f'{col} : Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
            st.markdown('---')

# --- Tab: Bivariée ---
with tabs[2]:
    st.header("🔄 Analyse Bivariée")
    all_cols = [col for info in var_dict.values() for col in info['cols'] if col in df.columns]
    x = st.selectbox('Variable X', all_cols)
    y = st.selectbox('Variable Y', all_cols, index=1)
    if x and y:
        fig = px.scatter(df, x=x, y=y, trendline='ols', title=f'Relation {x} vs {y}')
        st.plotly_chart(fig)

# --- Tab: Corrélations ---
with tabs[3]:
    st.header("🔗 Corrélations")
    dims = st.multiselect('Dimensions', list(var_dict.keys()), default=list(var_dict.keys())[:3])
    corr_cols = [f"{d}_{i}" for d in dims for i in range(1,5) if f"{d}_{i}" in df.columns]
    if corr_cols:
        corr = df[corr_cols].corr()
        fig = px.imshow(corr, title='Matrice de Corrélation')
        st.plotly_chart(fig, use_container_width=True)

# --- Tab: Préparation ---
with tabs[4]:
    st.header("🛠️ Préparation des Données")
    encoders = {}
    for col in ['Profil','Sexe','Diplome']:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_enc'] = le.fit_transform(df[col].fillna(''))
            encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    st.json(encoders)

# --- Tab: Modélisation ---
with tabs[5]:
    st.header("📈 Modélisation & Analyses Statistiques")
    model_type = st.selectbox("Type de modèle", ["Régression Linéaire", "Régression Logistique"])
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target = st.selectbox("Variable cible", num_cols)
    features = st.multiselect("Variables explicatives", [c for c in num_cols if c != target])
    if st.button("Entraîner le modèle"):
        if not features:
            st.error("Choisissez au moins une variable explicative.")
        else:
            X = df[features]
            y = df[target]
            if model_type == "Régression Linéaire":
                lr = LinearRegression()
                lr.fit(X, y)
                preds = lr.predict(X)
                st.metric("R² Score", f"{r2_score(y, preds):.3f}")
                coef_df = pd.DataFrame({"Variable": features, "Coefficient": lr.coef_}).set_index("Variable")
                st.bar_chart(coef_df)
            else:
                if y.nunique() != 2:
                    st.error("La variable cible doit être binaire pour la régression logistique.")
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                    log = LogisticRegression(max_iter=1000)
                    log.fit(X_train, y_train)
                    y_pred = log.predict(X_test)
                    y_proba = log.predict_proba(X_test)[:, 1]
                    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
                    st.metric("ROC AUC", f"{roc_auc_score(y_test, y_proba):.3f}")
                    st.text(classification_report(y_test, y_pred))

# --- Tab: Synthèse ---
with tabs[6]:
    st.header("🎯 Synthèse")
    scores = {info['name']: df[info['cols']].mean().mean() for info in var_dict.values() if set(info['cols']).issubset(df.columns)}
    categories = list(scores.keys())
    values = list(scores.values())
    fig = go.Figure(go.Scatterpolar(r=values + [values[0]], theta=categories + [categories[0]], fill='toself'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,5])))
    st.plotly_chart(fig, use_container_width=True)
    summary_df = pd.DataFrame(list(scores.items()), columns=['Dimension', 'Score'])
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        summary_df.to_excel(writer, index=False)
    st.download_button('Télécharger la synthèse', data=buf.getvalue(), file_name='synthese.xlsx')
