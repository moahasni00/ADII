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
from statsmodels.stats.outliers_influence import variance_inflation_factor
from io import BytesIO

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
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .stAlert > div {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stSuccess > div {
        background-color: #d4edda;
        color: #155724;
    }
    .stWarning > div {
        background-color: #fff3cd;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# Complete variable dictionary
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
    },
    'SAT': {
        'full_name': 'Satisfaction Utilisateur',
        'description': 'Évalue la satisfaction globale des utilisateurs',
        'items': {
            'SAT_1': 'Satisfaction générale',
            'SAT_2': 'Réponse aux besoins',
            'SAT_3': 'Facilité d\'utilisation',
            'SAT_4': 'Recommandation à d\'autres'
        }
    },
    'FOR': {
        'full_name': 'Formation et Support',
        'description': 'Évalue la qualité de la formation et du support',
        'items': {
            'FOR_1': 'Qualité des formations',
            'FOR_2': 'Disponibilité du support',
            'FOR_3': 'Clarté des guides',
            'FOR_4': 'Suivi post-formation'
        }
    },
    'RE': {
        'full_name': 'Résistance au Changement',
        'description': 'Mesure le niveau de résistance au changement',
        'items': {
            'RE_1': 'Appréhension au changement',
            'RE_2': 'Préférence anciens systèmes',
            'RE_3': 'Difficulté d\'adaptation',
            'RE_4': 'Stress lié au changement'
        }
    },
    'PE': {
        'full_name': 'Performance Attendue',
        'description': 'Évalue les attentes en termes de performance',
        'items': {
            'PE_1': 'Amélioration de la productivité',
            'PE_2': 'Efficacité accrue',
            'PE_3': 'Qualité du travail',
            'PE_4': 'Réduction des erreurs'
        }
    },
    'EE': {
        'full_name': 'Effort Attendu',
        'description': 'Mesure l\'effort perçu pour l\'utilisation',
        'items': {
            'EE_1': 'Facilité d\'apprentissage',
            'EE_2': 'Clarté d\'utilisation',
            'EE_3': 'Flexibilité',
            'EE_4': 'Maîtrise rapide'
        }
    },
    'FC': {
        'full_name': 'Conditions Facilitantes',
        'description': 'Évalue le support et les ressources disponibles',
        'items': {
            'FC_1': 'Support technique disponible',
            'FC_2': 'Ressources nécessaires',
            'FC_3': 'Compatibilité systèmes',
            'FC_4': 'Aide à disposition'
        }
    },
    'SI': {
        'full_name': 'Influence Sociale',
        'description': 'Mesure l\'impact de l\'environnement social',
        'items': {
            'SI_1': 'Encouragement des collègues',
            'SI_2': 'Support de la direction',
            'SI_3': 'Culture d\'innovation',
            'SI_4': 'Reconnaissance des efforts'
        }
    }
}

# Utility functions
@st.cache_data
def load_data():
    """Load and cache data with proper separator"""
    return pd.read_csv('Donn_es_simul_es_ADII.csv', sep=';')

def calculate_mean_score(dataframe, columns):
    """Calculate mean score with error handling"""
    try:
        return dataframe[columns].mean().mean()
    except KeyError as e:
        st.error(f"Erreur: Certaines colonnes ne sont pas trouvées dans le jeu de données. Vérifiez les noms des colonnes.")
        return None

def perform_advanced_analysis(data, var1, var2):
    """Perform advanced statistical analysis between two variables"""
    try:
        correlation = data[var1].corr(data[var2])
        mean_var1 = data[var1].mean()
        mean_var2 = data[var2].mean()
        std_var1 = data[var1].std()
        std_var2 = data[var2].std()
        t_stat, p_value = stats.ttest_ind(data[var1], data[var2])
        
        return {
            'correlation': correlation,
            'mean_var1': mean_var1,
            'mean_var2': mean_var2,
            'std_var1': std_var1,
            'std_var2': std_var2,
            't_stat': t_stat,
            'p_value': p_value
        }
    except Exception as e:
        st.error(f"Erreur lors de l'analyse statistique: {str(e)}")
        return None

def analyze_dimension(df, dimension_key):
    """Analyze a specific dimension and generate visualizations"""
    dim_cols = [f"{dimension_key}_{i}" for i in range(1, 5)]
    mean_scores = df[dim_cols].mean()
    
    fig = px.bar(
        x=dim_cols,
        y=mean_scores,
        title=f"Scores moyens - {var_dict[dimension_key]['full_name']}",
        labels={'x': 'Questions', 'y': 'Score Moyen'},
        color=mean_scores,
        color_continuous_scale='viridis'
    )
    
    return fig

def generate_insights(stats_results, var1_name, var2_name):
    """Generate insights based on statistical analysis results"""
    insights = []
    
    if stats_results['correlation'] > 0.7:
        insights.append(f"Forte corrélation positive entre {var1_name} et {var2_name}")
    elif stats_results['correlation'] < -0.7:
        insights.append(f"Forte corrélation négative entre {var1_name} et {var2_name}")
    
    if stats_results['p_value'] < 0.05:
        insights.append("Différence statistiquement significative entre les variables")
    else:
        insights.append("Pas de différence statistiquement significative")
    
    return insights

def analyze_trends(df, dimension):
    """Analyze trends over time for a specific dimension"""
    dim_cols = [f"{dimension}_{i}" for i in range(1, 5)]
    trend_data = df[dim_cols].mean(axis=1).rolling(window=10).mean()
    
    fig = px.line(
        trend_data,
        title=f"Évolution des scores - {var_dict[dimension]['full_name']}",
        labels={'value': 'Score Moyen', 'index': 'Observations'}
    )
    return fig

def compare_groups(df, dimension, group_col):
    """Compare dimension scores across different groups"""
    dim_cols = [f"{dimension}_{i}" for i in range(1, 5)]
    group_scores = df.groupby(group_col)[dim_cols].mean().mean(axis=1)
    
    fig = px.bar(
        group_scores,
        title=f"Comparaison par {group_col} - {var_dict[dimension]['full_name']}",
        labels={'value': 'Score Moyen', 'index': group_col}
    )
    return fig

def export_results(df, dimension_scores):
    """Export analysis results to various formats"""
    summary_df = pd.DataFrame(
        list(dimension_scores.items()),
        columns=['Dimension', 'Score Moyen']
    )
    
    export_format = st.selectbox(
        'Format d\'export',
        ['CSV', 'Excel', 'JSON']
    )
    
    if st.button('Exporter les résultats'):
        if export_format == 'CSV':
            csv = summary_df.to_csv(index=False)
            st.download_button(
                'Télécharger CSV',
                csv,
                'resultats_analyse.csv',
                'text/csv'
            )
        elif export_format == 'Excel':
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                summary_df.to_excel(writer, sheet_name='Résumé', index=False)
            st.download_button(
                'Télécharger Excel',
                output.getvalue(),
                'resultats_analyse.xlsx',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        else:
            json_str = summary_df.to_json(orient='records')
            st.download_button(
                'Télécharger JSON',
                json_str,
                'resultats_analyse.json',
                'application/json'
            )

# Load data
df = load_data()

# Sidebar navigation
st.sidebar.title('Navigation')

# Help section in sidebar
with st.sidebar:
    with st.expander('❓ Aide & Guide d\'Utilisation'):
        st.markdown("""
        ### 📚 Comment utiliser l'application:
        1. **Navigation**: Utilisez les onglets pour accéder aux différentes analyses
        2. **Interactivité**: Sélectionnez les variables et dimensions à analyser
        3. **Visualisation**: Explorez les graphiques interactifs
        4. **Export**: Téléchargez les résultats dans différents formats
        
        ### 🔍 Astuces:
        - Survolez les graphiques pour plus de détails
        - Utilisez les filtres pour affiner l'analyse
        - Consultez les interprétations pour chaque analyse
        """)

# Main content tabs
tabs = st.tabs([
    '📌 Accueil',
    '📊 Analyse Univariée',
    '🔄 Analyse Bivariée',
    '🔗 Corrélations',
    '🛠️ Préparation',
    '📐 Régression',
    '🎯 Synthèse'
])

# Tab 1: Home
with tabs[0]:
    st.title('📊 Tableau de Bord de la Transformation Digitale ADII')
    
    st.markdown("""
    ## 🎯 Projet : Application Streamlit – Analyse des données d'enquête sur la digitalisation à l'ADII
    
    Cette application permet d'analyser en détail les résultats de l'enquête sur la transformation digitale à l'ADII.
    Elle offre une vue complète des différentes dimensions évaluées et permet d'identifier les points forts et les axes
    d'amélioration.
    
    ### 🔑 Points Clés:
    - Analyse multidimensionnelle de la transformation digitale
    - Visualisations interactives des résultats
    - Recommandations personnalisées basées sur les données
    - Export des résultats dans différents formats
    """)
    
    st.markdown("### 🎯 Structure du Questionnaire")
    
    for var_key, var_info in var_dict.items():
        with st.expander(f"📌 {var_info['full_name']}"):
            st.markdown(f"**Description:** {var_info['description']}")
            
            var_cols = [f"{var_key}_{i}" for i in range(1, 5)]
            mean_score = calculate_mean_score(df, var_cols)
            
            if mean_score is not None:
                color = 'red' if mean_score < 2.5 else 'green' if mean_score > 3.5 else 'orange'
                st.markdown(f"**Score Moyen:** <span style='color:{color}'>{mean_score:.2f}/5</span>", unsafe_allow_html=True)
                
                st.markdown("#### Questions Associées:")
                for code, question in var_info['items'].items():
                    st.markdown(f"- **{code}:** {question}")
                
                try:
                    scores = df[var_cols].mean(axis=1)
                    fig = px.histogram(
                        scores,
                        title=f'Distribution des Scores - {var_info["full_name"]}',
                        labels={'value': 'Score Moyen', 'count': 'Fréquence'},
                        color_discrete_sequence=['#3498db'],
                        nbins=20
                    )
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
    
    with st.expander('👥 Analyse des Profils', expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(df, names='Profil', title='Répartition par Profil')
            st.plotly_chart(fig)
        
        with col2:
            st.markdown("""
            #### 💡 Interprétation
            - Visualisation de la distribution des profils
            - Identification des groupes majoritaires et minoritaires
            - Impact potentiel sur l'adoption digitale
            """)
    
    with st.expander('👨‍👩‍👧‍👦 Distribution par Sexe', expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                df,
                names='Sexe',
                title='Répartition par Sexe',
                color_discrete_sequence=['#FF69B4', '#4169E1']
            )
            st.plotly_chart(fig)
        
        with col2:
            st.markdown("""
            #### 💡 Interprétation
            - Analyse de la parité homme-femme
            - Implications pour les stratégies d'inclusion
            """)
    
    with st.expander('🎓 Niveau d\'Études', expanded=True):
        fig = px.bar(
            df['Diplome'].value_counts().reset_index(),
            x='index',
            y='Diplome',
            title='Distribution des Niveaux d\'Études',
            labels={'index': 'Diplôme', 'Diplome': 'Nombre'},
            color_discrete_sequence=['#2ecc71']
        )
        st.plotly_chart(fig)
        
        st.markdown("""
        #### 💡 Interprétation
        - Répartition des niveaux d'éducation
        - Corrélation potentielle avec l'adoption digitale
        """)

# Tab 3: Bivariate Analysis
with tabs[2]:
    st.header('🔄 Analyse Bivariée')
    
    col1, col2 = st.columns(2)
    with col1:
        var1 = st.selectbox(
            'Variable 1',
            [item for var in var_dict.values() for item in var['items'].keys()]
        )
    with col2:
        var2 = st.selectbox(
            'Variable 2',
            [item for var in var_dict.values() for item in var['items'].keys()],
            index=1
        )
    
    if var1 and var2:
        fig = px.scatter(
            df,
            x=var1,
            y=var2,
            title=f'Relation entre {var1} et {var2}',
            trendline='ols'
        )
        st.plotly_chart(fig)
        
        stats_results = perform_advanced_analysis(df, var1, var2)
        if stats_results:
            with st.expander('📊 Analyse Statistique Détaillée'):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric('Corrélation', f"{stats_results['correlation']:.3f}")
                with col2:
                    st.metric('P-value', f"{stats_results['p_value']:.3f}")
                with col3:
                    st.metric('T-statistic', f"{stats_results['t_stat']:.3f}")
                
                insights = generate_insights(stats_results, var1, var2)
                st.markdown("### 💡 Interprétation")
                for insight in insights:
                    st.markdown(f"- {insight}")

# Tab 4: Correlation Analysis
with tabs[3]:
    st.header('🔗 Analyse des Corrélations')
    
    dimensions = list(var_dict.keys())
    selected_dims = st.multiselect(
        'Sélectionnez les dimensions à analyser',
        dimensions,
        default=dimensions[:3]
    )
    
    if selected_dims:
        dim_cols = [f"{dim}_{i}" for dim in selected_dims for i in range(1, 5)]
        corr_matrix = df[dim_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            title='Matrice de Corrélation entre Dimensions',
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        fig.update_layout(height=600, margin=dict(t=50, b=0))
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander('📚 Guide d\'Interprétation des Corrélations'):
            st.markdown("""
            #### 🎯 Comment interpréter les corrélations:
            
            | Valeur | Interprétation |
            |---------|----------------|
            | 1.0 | Corrélation positive parfaite |
            | 0.7 à 0.9 | Corrélation positive forte |
            | 0.4 à 0.6 | Corrélation positive modérée |
            | 0.1 à 0.3 | Corrélation positive faible |
            | 0 | Aucune corrélation |
            | -0.3 à -0.1 | Corrélation négative faible |
            | -0.6 à -0.4 | Corrélation négative modérée |
            | -0.9 à -0.7 | Corrélation négative forte |
            | -1.0 | Corrélation négative parfaite |
            """)

# Tab 5: Data Preparation
with tabs[4]:
    st.header('🛠️ Préparation des Données')
    
    encoders = {}
    encoded_df = df.copy()
    
    with st.expander('🔄 Encodage des Variables Catégorielles', expanded=True):
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
    
    st.markdown("### 📊 Sélection des Variables")
    target_var = st.selectbox(
        "Variable à prédire",
        options=[item for var in var_dict.values() for item in var['items'].keys()]
    )
    
    features = st.multiselect(
        "Variables explicatives",
        options=[col for col in encoded_df.columns if col.endswith('_encoded')],
        default=[col for col in encoded_df.columns if col.endswith('_encoded')][:3]
    )
    
    if features and target_var:
        X = encoded_df[features]
        y = df[target_var]
        
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        with st.expander('📊 Résultats du Modèle', expanded=True):
            r2 = r2_score(y, y_pred)
            st.metric("R² Score", f"{r2:.3f}")
            
            coef_df = pd.DataFrame({
                'Variable': features,
                'Coefficient': model.coef_
            })
            
            fig = px.bar(
                coef_df,
                x='Variable',
                y='Coefficient',
                title='Coefficients de Régression',
                color='Coefficient',
                color_continuous_scale='RdBu'
            )
            st.plotly_chart(fig)
            
            st.markdown("""
            #### 💡 Interprétation
            - Les coefficients positifs indiquent une influence positive
            - Les coefficients négatifs indiquent une influence négative
            - L'amplitude indique la force de l'influence
            """)

# Tab 7: Synthesis
with tabs[6]:
    st.header('🎯 Synthèse et Recommandations')
    
    dimension_scores = {}
    for var_key, var_info in var_dict.items():
        var_cols = [f"{var_key}_{i}" for i in range(1, 5)]
        dimension_scores[var_info['full_name']] = df[var_cols].mean().mean()
    
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
            )
        ),
        showlegend=False,
        title='Vue d\'Ensemble de la Transformation Digitale'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander('📋 Recommandations Détaillées', expanded=True):
        strengths = {k: v for k, v in dimension_scores.items() if v >= 3.5}
        improvements = {k: v for k, v in dimension_scores.items() if v < 3.5}
        
        if strengths:
            st.markdown("### 💪 Points Forts")
            for dim, score in strengths.items():
                st.markdown(f"**{dim}** (Score: {score:.2f}/5)")
                st.markdown("- Maintenir les bonnes pratiques actuelles")
                st.markdown("- Partager les succès avec d'autres équipes")
        
        if improvements:
            st.markdown("### 🎯 Axes d'Amélioration")
            for dim, score in improvements.items():
                st.markdown(f"**{dim}** (Score: {score:.2f}/5)")
                st.markdown("- Identifier les obstacles spécifiques")
                st.markdown("- Mettre en place des formations ciblées")
                st.markdown("- Établir un plan d'action détaillé")
    
    st.markdown("### 📥 Export des Résultats")
    export_results(df, dimension_scores)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>📊 Tableau de Bord de la Transformation Digitale ADII | Version 1.0</p>
    <p><em>Développé avec Streamlit et Python</em></p>
    <p>Pour toute question ou suggestion: <a href='mailto:contact@adii.gov.ma'>contact@adii.gov.ma</a></p>
</div>
""", unsafe_allow_html=True)
