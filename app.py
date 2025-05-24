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

# Set page config
st.set_page_config(
    page_title="ADII Digital Transformation Analysis",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {padding: 0rem 1rem !important;}
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {font-size: 24px;}
    .reportview-container .main .block-container {padding-top: 2rem;}
    h1 {color: #1f77b4; text-align: center; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);}
    h2 {color: #2c3e50; margin-bottom: 1rem;}
    .stExpander {background-color: #f8f9fa; border-radius: 10px; margin-bottom: 1rem;}
    .explanation-box {background-color: #e8f4f8; padding: 1rem; border-radius: 5px; margin: 1rem 0;}
    .metric-card {background-color: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
</style>
""", unsafe_allow_html=True)

# Load and cache data
@st.cache_data
def load_data():
    df = pd.read_csv('Donn_es_simul_es_ADII.csv')
    return df

df = load_data()

# Complete variable dictionary
var_dict = {
    'ADT': {
        'full_name': 'Adoption de la Technologie',
        'description': 'Mesure le niveau d\'adoption des technologies numériques',
        'items': {
            'ADT_1': "J'utilise régulièrement les plateformes numériques mises en place",
            'ADT_2': "Ces outils font partie intégrante de mes tâches quotidiennes",
            'ADT_3': "Je réalise mes démarches via les systèmes digitaux de l'ADII",
            'ADT_4': "Je considère leur usage comme indispensable"
        }
    },
    'INT': {
        'full_name': 'Intention d\'Utilisation',
        'description': 'Évalue l\'intention future d\'utilisation des outils numériques',
        'items': {
            'INT_1': "J'ai l'intention de continuer à utiliser les outils digitaux",
            'INT_2': "Je prévois d'utiliser davantage les services numériques",
            'INT_3': "Je recommande l'utilisation des plateformes digitales",
            'INT_4': "Je compte approfondir ma maîtrise des outils numériques"
        }
    },
    'SAT': {
        'full_name': 'Satisfaction',
        'description': 'Mesure la satisfaction globale envers les outils numériques',
        'items': {
            'SAT_1': "Je suis satisfait(e) des fonctionnalités offertes",
            'SAT_2': "Les outils répondent à mes besoins professionnels",
            'SAT_3': "L'utilisation des plateformes améliore mon efficacité",
            'SAT_4': "Je suis content(e) de travailler avec ces outils"
        }
    },
    'FOR': {
        'full_name': 'Formation',
        'description': 'Évalue la qualité et l\'impact de la formation reçue',
        'items': {
            'FOR_1': "J'ai reçu une formation adéquate",
            'FOR_2': "La formation m'a permis de bien utiliser les outils",
            'FOR_3': "Le support technique est disponible si besoin",
            'FOR_4': "Les ressources d'apprentissage sont accessibles"
        }
    },
    'RE': {
        'full_name': 'Résistance au Changement',
        'description': 'Mesure le niveau de résistance face au changement digital',
        'items': {
            'RE_1': "J'ai des difficultés à m'adapter aux nouveaux outils",
            'RE_2': "Je préfère les méthodes de travail traditionnelles",
            'RE_3': "Le changement digital me stresse",
            'RE_4': "J'ai besoin de plus de temps pour m'adapter"
        }
    },
    'PE': {
        'full_name': 'Performance Attendue',
        'description': 'Évalue les attentes en termes de performance',
        'items': {
            'PE_1': "Les outils améliorent ma productivité",
            'PE_2': "La digitalisation facilite mon travail",
            'PE_3': "Les plateformes sont utiles pour mes tâches",
            'PE_4': "L'utilisation des outils optimise mon temps"
        }
    },
    'EE': {
        'full_name': 'Effort Attendu',
        'description': 'Mesure l\'effort perçu pour utiliser les outils',
        'items': {
            'EE_1': "Les outils sont faciles à utiliser",
            'EE_2': "L'interface est claire et intuitive",
            'EE_3': "Je comprends facilement les fonctionnalités",
            'EE_4': "L'utilisation ne demande pas trop d'effort"
        }
    },
    'FC': {
        'full_name': 'Conditions Facilitantes',
        'description': 'Évalue les conditions qui facilitent l\'utilisation',
        'items': {
            'FC_1': "J'ai les ressources nécessaires",
            'FC_2': "L'infrastructure technique est adéquate",
            'FC_3': "Le support est disponible en cas de besoin",
            'FC_4': "L'environnement favorise l'utilisation"
        }
    },
    'SI': {
        'full_name': 'Influence Sociale',
        'description': 'Mesure l\'impact de l\'environnement social',
        'items': {
            'SI_1': "Mes collègues encouragent l'utilisation",
            'SI_2': "La direction soutient la transformation digitale",
            'SI_3': "L'utilisation est valorisée dans l'organisation",
            'SI_4': "Mon entourage professionnel utilise ces outils"
        }
    }
}

# Sidebar for navigation
st.sidebar.title('🧭 Navigation')
st.sidebar.markdown('---')

# Main content
st.title('📊 Analyse de la Transformation Digitale ADII')
st.markdown('---')

# Create tabs
tabs = st.tabs([
    "🏠 Accueil",
    "📈 Analyse Univariée",
    "🔄 Analyse Bivariée",
    "🔗 Corrélations",
    "🔧 Préparation",
    "📐 Régression",
    "🎯 Synthèse"
])

# Tab 1: Home and Variable Understanding
with tabs[0]:
    st.header("📚 Structure du Questionnaire")
    
    # Introduction box
    with st.expander("ℹ️ Guide de Lecture", expanded=True):
        st.markdown("""
        ### 📋 Échelle de Likert utilisée:
        1. 🔴 Pas du tout d'accord
        2. 🟠 Pas d'accord
        3. 🟡 Ni d'accord, ni pas d'accord
        4. 🟢 D'accord
        5. 🔵 Tout à fait d'accord
        
        ### 📊 Interprétation des scores:
        - **Score < 2.5**: Perception négative
        - **Score 2.5-3.5**: Perception neutre
        - **Score > 3.5**: Perception positive
        """)
    
    # Display variables with interactive elements
    for var, details in var_dict.items():
        with st.expander(f"📌 {var} - {details['full_name']}"):
            st.markdown(f"### Description")
            st.info(details['description'])
            
            # Calculate mean score for the variable
            var_cols = [f"{var}_{i}" for i in range(1, 5)]
            mean_score = df[var_cols].mean().mean()
            
            # Display mean score with color coding
            score_color = 'red' if mean_score < 2.5 else 'green' if mean_score > 3.5 else 'orange'
            st.markdown(f"### Score moyen: <span style='color:{score_color}'>{mean_score:.2f}</span>/5", unsafe_allow_html=True)
            
            # Display items with interactive elements
            for item, question in details['items'].items():
                st.markdown(f"#### {item}")
                st.markdown(f"*{question}*")
                
                # Create interactive visualization
                fig = px.histogram(df, x=item,
                                 title=f"Distribution des réponses - {item}",
                                 color=item,
                                 color_discrete_sequence=px.colors.sequential.Viridis,
                                 nbins=5)
                
                fig.update_layout(
                    showlegend=False,
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add statistics in an expandable section
                with st.expander("📊 Statistiques détaillées"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Moyenne", f"{df[item].mean():.2f}")
                    with col2:
                        st.metric("Médiane", f"{df[item].median():.2f}")
                    with col3:
                        st.metric("Écart-type", f"{df[item].std():.2f}")
                    
                    # Add interpretation
                    mean_val = df[item].mean()
                    interpretation = """
                    #### 💡 Interprétation:
                    """
                    if mean_val < 2.5:
                        interpretation += "Les répondants montrent une tendance négative sur cet aspect."
                    elif mean_val > 3.5:
                        interpretation += "Les répondants montrent une tendance positive sur cet aspect."
                    else:
                        interpretation += "Les répondants ont une opinion mitigée sur cet aspect."
                    
                    st.markdown(interpretation)

# Tab 2: Univariate Analysis
with tabs[1]:
    st.header("📊 Analyse Descriptive Univariée")
    
    # Profile Analysis
    with st.expander("👥 Analyse des Profils", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Profile distribution
            fig_profile = px.pie(df, names='Profil',
                               title='Répartition des Profils',
                               color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig_profile)
            
        with col2:
            # Interpretation box
            st.markdown("""
            ### 💡 Interprétation de la distribution des profils
            
            Cette visualisation montre la répartition des différents profils dans l'organisation.
            Une distribution équilibrée suggère une bonne représentativité de l'échantillon.
            """)
            
            # Add profile statistics
            profile_counts = df['Profil'].value_counts()
            for profile, count in profile_counts.items():
                st.metric(profile, f"{count} ({count/len(df)*100:.1f}%)"))
    
    # Gender Analysis
    with st.expander("👫 Analyse par Genre"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Gender distribution
            fig_gender = px.pie(df, names='Sexe',
                              title='Répartition par Genre',
                              color_discrete_sequence=['#FF9999', '#66B2FF'])
            st.plotly_chart(fig_gender)
            
        with col2:
            st.markdown("""
            ### 💡 Interprétation de la distribution par genre
            
            Cette visualisation permet d'analyser la représentation des genres dans l'échantillon
            et d'identifier d'éventuelles disparités.
            """)
            
            # Add gender statistics
            gender_counts = df['Sexe'].value_counts()
            for gender, count in gender_counts.items():
                st.metric(gender, f"{count} ({count/len(df)*100:.1f}%)"))
    
    # Education Analysis
    with st.expander("🎓 Analyse des Diplômes"):
        # Education distribution
        fig_edu = px.bar(df['Diplome'].value_counts().reset_index(),
                        x='index', y='Diplome',
                        title='Distribution des Niveaux d\'Éducation',
                        labels={'index': 'Diplôme', 'Diplome': 'Nombre'},
                        color='Diplome',
                        color_continuous_scale='Viridis')
        st.plotly_chart(fig_edu)
        
        with st.expander("💡 Interprétation"):
            st.markdown("""
            ### Analyse du niveau d'éducation
            
            Cette distribution nous permet de comprendre le niveau de qualification
            des répondants et son potentiel impact sur l'adoption numérique.
            """)

# Tab 3: Bivariate Analysis
with tabs[2]:
    st.header("🔄 Analyse Bivariée")
    
    # Variable selection
    col1, col2 = st.columns(2)
    with col1:
        selected_var = st.selectbox(
            "Sélectionnez une variable à analyser",
            options=[item for var in var_dict.values() for item in var['items'].keys()]
        )
    with col2:
        group_var = st.selectbox(
            "Grouper par",
            options=['Profil', 'Sexe', 'Diplome']
        )
    
    # Create visualization
    fig = px.box(df, x=group_var, y=selected_var,
                 color=group_var,
                 title=f'Distribution de {selected_var} par {group_var}')
    st.plotly_chart(fig)
    
    # Statistical test
    with st.expander("📊 Analyse Statistique"):
        if len(df[group_var].unique()) == 2:
            # T-test for two groups
            group1, group2 = df[group_var].unique()
            stat, pval = stats.ttest_ind(
                df[df[group_var] == group1][selected_var],
                df[df[group_var] == group2][selected_var]
            )
            test_name = "Test t de Student"
        else:
            # ANOVA for more than two groups
            groups = [group for name, group in df.groupby(group_var)[selected_var]]
            stat, pval = stats.f_oneway(*groups)
            test_name = "ANOVA"
        
        st.markdown(f"### Résultats du {test_name}")
        st.metric("Statistique de test", f"{stat:.4f}")
        st.metric("P-value", f"{pval:.4f}")
        
        # Interpretation
        st.markdown("### 💡 Interprétation")
        if pval < 0.05:
            st.success(f"Il existe une différence significative entre les groupes (p < 0.05)")
        else:
            st.info(f"Pas de différence significative entre les groupes (p > 0.05)")

# Tab 4: Correlation Analysis
with tabs[3]:
    st.header("🔗 Analyse des Corrélations")
    
    # Get numeric columns for correlation
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = df[numeric_cols].corr()
    
    # Create heatmap
    fig = px.imshow(correlation_matrix,
                    labels=dict(color="Corrélation"),
                    color_continuous_scale="RdBu",
                    title="Matrice de Corrélation")
    
    st.plotly_chart(fig)
    
    with st.expander("💡 Interprétation des Corrélations"):
        st.markdown("""
        ### Guide d'interprétation:
        - **Corrélation positive forte** (> 0.7): Variables qui évoluent fortement dans le même sens
        - **Corrélation positive modérée** (0.3 - 0.7): Variables qui évoluent modérément dans le même sens
        - **Corrélation faible** (-0.3 - 0.3): Peu ou pas de relation linéaire
        - **Corrélation négative modérée** (-0.7 - -0.3): Variables qui évoluent modérément en sens inverse
        - **Corrélation négative forte** (< -0.7): Variables qui évoluent fortement en sens inverse
        """)
        
        # Find strongest correlations
        correlations = correlation_matrix.unstack()
        sorted_correlations = correlations[correlations != 1.0].sort_values(ascending=False)
        
        st.markdown("### 🔍 Corrélations les plus fortes:")
        for idx, value in sorted_correlations[:5].items():
            st.markdown(f"- **{idx[0]}** et **{idx[1]}**: {value:.3f}")

# Tab 5: Data Preparation
with tabs[4]:
    st.header("🔧 Préparation des Données")
    
    # Encoding categorical variables
    with st.expander("🔄 Encodage des Variables Catégorielles"):
        st.markdown("### Variables encodées:")
        
        encoders = {}
        encoded_df = df.copy()
        
        for col in ['Profil', 'Sexe', 'Diplome']:
            le = LabelEncoder()
            encoded_df[f'{col}_encoded'] = le.fit_transform(df[col])
            encoders[col] = le
            
            # Display encoding mapping
            st.markdown(f"#### {col}:")
            for i, label in enumerate(le.classes_):
                st.markdown(f"- {label}: {i}")
    
    # Target variable selection
    st.markdown("### 🎯 Sélection de la Variable Cible")
    target_var = st.selectbox(
        "Choisissez la variable à prédire",
        options=[item for var in var_dict.values() for item in var['items'].keys()]
    )
    
    # Display target variable distribution
    fig = px.histogram(df, x=target_var,
                       title=f'Distribution de la Variable Cible: {target_var}',
                       color_discrete_sequence=['#3498db'])
    st.plotly_chart(fig)
    
    with st.expander("💡 Analyse de la Distribution"):
        st.markdown(f"### Statistiques de {target_var}")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Moyenne", f"{df[target_var].mean():.2f}")
        with col2:
            st.metric("Médiane", f"{df[target_var].median():.2f}")
        with col3:
            st.metric("Écart-type", f"{df[target_var].std():.2f}")

# Tab 6: Linear Regression
with tabs[5]:
    st.header("📐 Régression Linéaire")
    
    # Feature selection
    st.markdown("### 📊 Sélection des Variables Explicatives")
    features = st.multiselect(
        "Choisissez les variables explicatives",
        options=[col for col in encoded_df.columns if col.endswith('_encoded')],
        default=[col for col in encoded_df.columns if col.endswith('_encoded')][:3]
    )
    
    if features and target_var:
        # Prepare data
        X = encoded_df[features]
        y = df[target_var]
        
        # Fit model
        model = LinearRegression()
        model.fit(X, y)
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Model results
        with st.expander("📊 Résultats du Modèle", expanded=True):
            # R² score
            r2 = r2_score(y, y_pred)
            st.metric("R² Score", f"{r2:.3f}")
            
            # Coefficients
            st.markdown("### Coefficients:")
            coef_df = pd.DataFrame({
                'Variable': features,
                'Coefficient': model.coef_
            })
            
            fig_coef = px.bar(coef_df, x='Variable', y='Coefficient',
                              title='Coefficients de Régression',
                              color='Coefficient',
                              color_continuous_scale='RdBu')
            st.plotly_chart(fig_coef)
            
            # VIF Analysis
            st.markdown("### Analyse de Multicolinéarité (VIF)")
            X_vif = pd.DataFrame(X, columns=features)
            vif_data = pd.DataFrame()
            vif_data["Variable"] = X_vif.columns
            vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) 
                               for i in range(X_vif.shape[1])]
            
            fig_vif = px.bar(vif_data, x='Variable', y='VIF',
                             title='Facteurs d\'Inflation de la Variance (VIF)',
                             color='VIF',
                             color_continuous_scale='Viridis')
            st.plotly_chart(fig_vif)
            
            with st.expander("💡 Interprétation du VIF"):
                st.markdown("""
                ### Guide d'interprétation du VIF:
                - **VIF < 5**: Pas de multicolinéarité problématique
                - **5 < VIF < 10**: Multicolinéarité modérée
                - **VIF > 10**: Forte multicolinéarité, problématique
                """)
        
        # Residual analysis
        with st.expander("📉 Analyse des Résidus"):
            residuals = y - y_pred
            
            # Residuals vs Predicted
            fig_resid = px.scatter(x=y_pred, y=residuals,
                                 labels={'x': 'Valeurs Prédites', 'y': 'Résidus'},
                                 title='Résidus vs Valeurs Prédites')
            fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_resid)
            
            # Residuals distribution
            fig_resid_dist = px.histogram(residuals,
                                        title='Distribution des Résidus',
                                        color_discrete_sequence=['#3498db'])
            st.plotly_chart(fig_resid_dist)

# Tab 7: Final Visualization
with tabs[6]:
    st.header("🎯 Synthèse des Résultats")
    
    # Overall scores by dimension
    st.markdown("### 📊 Scores Moyens par Dimension")
    
    # Calculate mean scores for each dimension
    dimension_scores = {}
    for var in var_dict.keys():
        var_cols = [f"{var}_{i}" for i in range(1, 5)]
        dimension_scores[var_dict[var]['full_name']] = df[var_cols].mean().mean()
    
    # Create radar chart
    categories = list(dimension_scores.keys())
    values = list(dimension_scores.values())
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],  # Repeat first value to close the polygon
        theta=categories + [categories[0]],  # Repeat first category to close the polygon
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
    
    with st.expander("💡 Interprétation Globale"):
        st.markdown("""
        ### Analyse des Dimensions:
        
        Cette visualisation permet d'identifier:
        - Les dimensions les plus développées
        - Les axes d'amélioration prioritaires
        - L'équilibre global de la transformation digitale
        """)
        
        # Display metrics for each dimension
        for dim, score in dimension_scores.items():
            color = 'red' if score < 2.5 else 'green' if score > 3.5 else 'orange'
            st.markdown(f"**{dim}**: <span style='color:{color}'>{score:.2f}/5</span>", unsafe_allow_html=True)
    
    # Final recommendations
    with st.expander("📋 Recommandations"):
        st.markdown("### Points Clés et Recommandations")
        
        # Identify strengths and weaknesses
        strengths = {k: v for k, v in dimension_scores.items() if v > 3.5}
        weaknesses = {k: v for k, v in dimension_scores.items() if v < 2.5}
        
        if strengths:
            st.markdown("#### 💪 Points Forts:")
            for dim, score in strengths.items():
                st.markdown(f"- **{dim}**: {score:.2f}/5")
        
        if weaknesses:
            st.markdown("#### 🎯 Axes d'Amélioration:")
            for dim, score in weaknesses.items():
                st.markdown(f"- **{dim}**: {score:.2f}/5")
        
        st.markdown("#### 📈 Suggestions d'Actions:")
        for dim, score in weaknesses.items():
            st.markdown(f"- Renforcer {dim.lower()} par des actions ciblées")

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>📊 Tableau de Bord de la Transformation Digitale ADII</p>
</div>
""", unsafe_allow_html=True)
