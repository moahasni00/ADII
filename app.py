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
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Page configuration
st.set_page_config(page_title="ADII Digital Transformation Dashboard", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
        padding: 1rem;
    }
    .st-emotion-cache-16idsys p {
        font-size: 1.1rem;
        line-height: 1.6;
    }
    .st-emotion-cache-16idsys h1 {
        color: #1f77b4;
    }
    .st-emotion-cache-16idsys h2, .st-emotion-cache-16idsys h3 {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# Variable dictionary with exact column names from CSV
var_dict = {
    'ADT': {
        'full_name': 'Adoption de la Transformation Digitale',
        'description': 'Mesure le niveau d\'adoption des outils numériques',
        'items': {
            'ADT_1': 'Utilisation régulière des outils numériques',
            'ADT_2': 'Intégration dans le travail quotidien',
            'ADT_3': 'Adaptation aux nouveaux outils',
            'ADT_4': 'Participation aux formations digitales'
        }
    },
    'INT': {
        'full_name': 'Intention d\'Utilisation',
        'description': 'Évalue la volonté d\'utiliser les outils numériques',
        'items': {
            'INT_1': 'Intention d\'utilisation future',
            'INT_2': 'Planification d\'utilisation',
            'INT_3': 'Recommandation aux collègues',
            'INT_4': 'Engagement dans la transformation'
        }
    }
    # Add other variables similarly
}

# Sidebar
st.sidebar.title('Navigation')
tabs = st.tabs(['📌 Accueil', '📊 Analyse Univariée', '🔄 Analyse Bivariée', 
                '🔗 Corrélations', '🛠️ Préparation', '📐 Régression', '🎯 Synthèse'])

# Tab 1: Home
with tabs[0]:
    st.title('📊 Tableau de Bord de la Transformation Digitale ADII')
    
    st.markdown("### 🎯 Structure du Questionnaire")
    
    for var_key, var_info in var_dict.items():
        with st.expander(f"📌 {var_info['full_name']}"):
            st.markdown(f"**Description:** {var_info['description']}")
            
            # Calculate mean score with error handling
            var_cols = [f"{var_key}_{i}" for i in range(1, 5)]
            mean_score = calculate_mean_score(df, var_cols)
            
            if mean_score is not None:
                # Add color coding based on score
                color = 'red' if mean_score < 2.5 else 'green' if mean_score > 3.5 else 'orange'
                st.markdown(f"**Score Moyen:** <span style='color:{color}'>{mean_score:.2f}/5</span>", unsafe_allow_html=True)
                
                # Display items in a more attractive table
                st.markdown("#### Questions Associées:")
                for code, question in var_info['items'].items():
                    st.markdown(f"- **{code}:** {question}")
                
                # Show distribution with improved styling
                try:
                    scores = df[var_cols].mean(axis=1)
                    fig = px.histogram(scores,
                                     title=f'Distribution des Scores - {var_info["full_name"]}',
                                     labels={'value': 'Score Moyen', 'count': 'Fréquence'},
                                     color_discrete_sequence=['#3498db'],
                                     nbins=20)
                    fig.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        margin=dict(t=40, l=0, r=0, b=0)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error("Erreur lors de la création du graphique.")

# Tab 2: Univariate Analysis
with tabs[1]:
    st.header('📊 Analyse Descriptive Univariée')
    
    # Profile analysis
    with st.expander('👥 Analyse des Profils', expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(df, names='Profil', title='Répartition par Profil')
            st.plotly_chart(fig)
        
        with col2:
            st.markdown("""#### 💡 Interprétation
            - Visualisation de la distribution des profils
            - Identification des groupes majoritaires et minoritaires
            - Impact potentiel sur l'adoption digitale""")
    
    # Gender analysis
    with st.expander('👨‍👩‍👧‍👦 Distribution par Sexe', expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(df, names='Sexe', title='Répartition par Sexe',
                        color_discrete_sequence=['#FF69B4', '#4169E1'])
            st.plotly_chart(fig)
        
        with col2:
            st.markdown("""#### 💡 Interprétation
            - Analyse de la parité homme-femme
            - Implications pour les stratégies d'inclusion""")
    
    # Education analysis
    with st.expander('🎓 Niveau d\'Études', expanded=True):
        fig = px.bar(df['Diplome'].value_counts().reset_index(),
                    x='index', y='Diplome',
                    title='Distribution des Niveaux d\'Études',
                    labels={'index': 'Diplôme', 'Diplome': 'Nombre'},
                    color_discrete_sequence=['#2ecc71'])
        st.plotly_chart(fig)
        
        st.markdown("""#### 💡 Interprétation
        - Répartition des niveaux d'éducation
        - Corrélation potentielle avec l'adoption digitale""")

# Tab 3: Bivariate Analysis
with tabs[2]:
    st.header('🔄 Analyse Bivariée')
    
    col1, col2 = st.columns(2)
    with col1:
        var1 = st.selectbox('Variable 1',
                           [item for var in var_dict.values() 
                            for item in var['items'].keys()])
    with col2:
        var2 = st.selectbox('Variable 2',
                           [item for var in var_dict.values() 
                            for item in var['items'].keys()],
                           index=1)
    
    if var1 and var2:
        # Scatter plot
        fig = px.scatter(df, x=var1, y=var2,
                        title=f'Relation entre {var1} et {var2}',
                        trendline='ols')
        st.plotly_chart(fig)
        
        # Statistical test
        correlation = df[var1].corr(df[var2])
        with st.expander('📊 Analyse Statistique'):
            st.metric('Coefficient de Corrélation', f"{correlation:.3f}")
            
            st.markdown(f"""#### 💡 Interprétation
            - Corrélation {'positive' if correlation > 0 else 'négative'}
            - Force de la relation: {'forte' if abs(correlation) > 0.7 
                                    else 'modérée' if abs(correlation) > 0.3 
                                    else 'faible'}
            """)

# Tab 4: Correlation Analysis
with tabs[3]:
    st.header('🔗 Analyse des Corrélations')
    
    # Select variables for correlation
    numeric_cols = [col for col in df.columns 
                   if col.startswith(tuple(var_dict.keys()))]
    
    with st.expander('🔍 Matrice de Corrélation', expanded=True):
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix,
                        title='Matrice de Corrélation',
                        color_continuous_scale='RdBu_r')
        st.plotly_chart(fig)
        
        with st.expander('💡 Guide d\'Interprétation'):
            st.markdown("""
            #### Comment lire la matrice:
            - **Rouge**: Corrélation positive forte
            - **Blanc**: Pas de corrélation
            - **Bleu**: Corrélation négative forte
            
            #### Points clés:
            - Identifier les variables fortement corrélées
            - Détecter les patterns de relations
            - Éviter la multicolinéarité dans les analyses""")

# Tab 5: Data Preparation
with tabs[4]:
    st.header('🛠️ Préparation des Données')
    
    # Initialize encoders dictionary
    encoders = {}
    
    with st.expander('🔄 Encodage des Variables Catégorielles', expanded=True):
        encoded_df = df.copy()
        
        for col in ['Profil', 'Sexe', 'Diplome']:
            le = LabelEncoder()
            encoded_df[f'{col}_encoded'] = le.fit_transform(df[col])
            encoders[col] = le
            
            st.markdown(f"#### {col}:")
            for i, label in enumerate(le.classes_):
                st.markdown(f"- {label}: {i}")

# Tab 6: Linear Regression
with tabs[5]:
    st.header('📐 Régression Linéaire')
    
    # Feature selection
    st.markdown("### 📊 Sélection des Variables")
    target_var = st.selectbox(
        "Variable à prédire",
        options=[item for var in var_dict.values() 
                for item in var['items'].keys()]
    )
    
    features = st.multiselect(
        "Variables explicatives",
        options=[col for col in encoded_df.columns 
                if col.endswith('_encoded')],
        default=[col for col in encoded_df.columns 
                if col.endswith('_encoded')][:3]
    )
    
    if features and target_var:
        # Prepare data
        X = encoded_df[features]
        y = df[target_var]
        
        # Fit model
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Model results
        with st.expander('📊 Résultats du Modèle', expanded=True):
            # R² score
            r2 = r2_score(y, y_pred)
            st.metric("R² Score", f"{r2:.3f}")
            
            # Coefficients visualization
            coef_df = pd.DataFrame({
                'Variable': features,
                'Coefficient': model.coef_
            })
            
            fig = px.bar(coef_df, x='Variable', y='Coefficient',
                         title='Coefficients de Régression',
                         color='Coefficient',
                         color_continuous_scale='RdBu')
            st.plotly_chart(fig)
            
            st.markdown("""#### 💡 Interprétation
            - Les coefficients positifs indiquent une influence positive
            - Les coefficients négatifs indiquent une influence négative
            - L'amplitude indique la force de l'influence""")

# Tab 7: Final Visualization
with tabs[6]:
    st.header('🎯 Synthèse des Résultats')
    
    # Calculate dimension scores
    dimension_scores = {}
    for var_key, var_info in var_dict.items():
        var_cols = [f"{var_key}_{i}" for i in range(1, 5)]
        dimension_scores[var_info['full_name']] = df[var_cols].mean().mean()
    
    # Radar chart
    categories = list(dimension_scores.keys())
    values = list(dimension_scores.values())
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Scores Moyens'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5]
            )),
        showlegend=False,
        title='Radar des Dimensions de la Transformation Digitale'
    )
    
    st.plotly_chart(fig)
    
    # Recommendations
    with st.expander('📋 Recommandations', expanded=True):
        strengths = {k: v for k, v in dimension_scores.items() if v > 3.5}
        weaknesses = {k: v for k, v in dimension_scores.items() if v < 2.5}
        
        if strengths:
            st.markdown("#### 💪 Points Forts:")
            for dim, score in strengths.items():
                st.markdown(f"- **{dim}**: {score:.2f}/5")
        
        if weaknesses:
            st.markdown("#### 🎯 Axes d'Amélioration:")
            for dim, score in weaknesses.items():
                st.markdown(f"- **{dim}**: {score:.2f}/5 - Renforcer par des actions ciblées")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>📊 Tableau de Bord de la Transformation Digitale ADII</p>
    <p><em>Développé avec Streamlit et Python</em></p>
</div>
""", unsafe_allow_html=True)

# Load and cache data with proper separator
@st.cache_data
def load_data():
    return pd.read_csv('Donn_es_simul_es_ADII.csv', sep=';')

df = load_data()

# Add error handling for mean score calculation
def calculate_mean_score(dataframe, columns):
    try:
        return dataframe[columns].mean().mean()
    except KeyError as e:
        st.error(f"Erreur: Certaines colonnes ne sont pas trouvées dans le jeu de données. Vérifiez les noms des colonnes.")
        return None
