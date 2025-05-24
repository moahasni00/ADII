import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.express as px
import ast

# Configuration de la page
st.set_page_config(page_title="Analyse ADII", layout="wide")

# Fonction pour calculer les moyennes des scores
def calculate_means(df):
    # Calculer les moyennes pour chaque groupe de variables
    score_columns = {
        'ADT': [col for col in df.columns if col.startswith('ADT_')],
        'INT': [col for col in df.columns if col.startswith('INT_')],
        'SAT': [col for col in df.columns if col.startswith('SAT_')],
        'FOR': [col for col in df.columns if col.startswith('FOR_')],
        'RE': [col for col in df.columns if col.startswith('RE_')],
        'PE': [col for col in df.columns if col.startswith('PE_')],
        'EE': [col for col in df.columns if col.startswith('EE_')],
        'FC': [col for col in df.columns if col.startswith('FC_')],
        'SI': [col for col in df.columns if col.startswith('SI_')]
    }
    
    for score_name, cols in score_columns.items():
        df[score_name] = df[cols].mean(axis=1)
    
    return df

# Chargement des données
@st.cache_data
def load_data():
    df = pd.read_csv('Donn_es_simul_es_ADII.csv', sep=';')
    # Convertir la colonne Plateformes_utilisees de string à liste
    df['Plateformes_utilisees'] = df['Plateformes_utilisees'].apply(ast.literal_eval)
    return calculate_means(df)

df = load_data()

# Sidebar
st.sidebar.title('Navigation')
page = st.sidebar.radio(
    'Sélectionnez une page',
    ['Analyse univariée', 'Analyse bivariée', 'Corrélations', 
     'Préparation des données', 'Régression linéaire', 'Visualisation finale']
)

# Page 1: Analyse univariée
if page == 'Analyse univariée':
    st.title('📊 Analyse descriptive univariée')
    
    # Variables quantitatives
    st.header('Statistiques descriptives des variables quantitatives')
    score_vars = ['ADT', 'INT', 'SAT', 'FOR', 'RE', 'PE', 'EE', 'FC', 'SI']
    stats_df = df[score_vars].describe()
    st.dataframe(stats_df)
    
    # Visualisations
    col1, col2 = st.columns(2)
    
    # Histogrammes
    with col1:
        st.subheader('Histogrammes des scores')
        selected_var = st.selectbox('Sélectionnez une variable', score_vars)
        fig = plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=selected_var, kde=True)
        st.pyplot(fig)
    
    # Boxplots
    with col2:
        st.subheader('Boxplots des scores')
        fig = plt.figure(figsize=(12, 6))
        sns.boxplot(data=df[score_vars])
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    # Variables qualitatives
    st.header('Répartition des variables qualitatives')
    qual_cols = ['Profil', 'Sexe', 'Diplome']
    
    cols = st.columns(len(qual_cols))
    for i, col in enumerate(qual_cols):
        with cols[i]:
            st.subheader(f'Distribution de {col}')
            fig = plt.figure(figsize=(8, 6))
            df[col].value_counts().plot(kind='bar')
            plt.xticks(rotation=45)
            st.pyplot(fig)
    
    # Plateformes utilisées
    st.subheader('Fréquence d\'utilisation des plateformes')
    platforms = [item for sublist in df['Plateformes_utilisees'] for item in sublist]
    platform_counts = pd.Series(platforms).value_counts()
    fig = plt.figure(figsize=(12, 6))
    platform_counts.plot(kind='bar')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Page 2: Analyse bivariée
elif page == 'Analyse bivariée':
    st.title('🔄 Analyse bivariée')
    
    score_vars = ['ADT', 'INT', 'SAT', 'FOR', 'RE', 'PE', 'EE', 'FC', 'SI']
    cat_var = st.selectbox('Sélectionnez une variable catégorielle', ['Profil', 'Sexe', 'Diplome'])
    score_var = st.selectbox('Sélectionnez une variable quantitative', score_vars)
    
    # Comparaison des moyennes
    st.subheader('Comparaison des moyennes par groupe')
    means = df.groupby(cat_var)[score_var].mean()
    st.dataframe(means)
    
    # Boxplot comparatif
    st.subheader('Boxplot comparatif')
    fig = plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=cat_var, y=score_var)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Test statistique
    st.subheader('Test statistique')
    groups = [group for _, group in df.groupby(cat_var)[score_var]]
    
    # Test de normalité
    _, p_norm = stats.normaltest(df[score_var])
    
    if p_norm < 0.05:
        # Utiliser Kruskal-Wallis si non normal
        h_stat, p_val = stats.kruskal(*groups)
        test_name = 'Kruskal-Wallis'
    else:
        # Utiliser ANOVA si normal
        f_stat, p_val = stats.f_oneway(*groups)
        test_name = 'ANOVA'
    
    st.write(f'Résultat du test {test_name}: p-value = {p_val:.4f}')
    if p_val < 0.05:
        st.write('Il existe des différences significatives entre les groupes.')
    else:
        st.write('Pas de différences significatives entre les groupes.')

# Page 3: Corrélations
elif page == 'Corrélations':
    st.title('🔗 Analyse des corrélations')
    
    score_vars = ['ADT', 'INT', 'SAT', 'FOR', 'RE', 'PE', 'EE', 'FC', 'SI']
    corr_matrix = df[score_vars].corr()
    
    # Heatmap
    st.subheader('Matrice de corrélation')
    fig = plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    st.pyplot(fig)
    
    # Corrélations fortes
    st.subheader('Corrélations fortes (|r| > 0.7)')
    strong_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                strong_corr.append(f'{corr_matrix.columns[i]} - {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.3f}')
    
    if strong_corr:
        for corr in strong_corr:
            st.write(corr)
    else:
        st.write('Aucune corrélation forte trouvée')

# Page 4: Préparation des données
elif page == 'Préparation des données':
    st.title('🔧 Préparation des données')
    
    # Encodage des variables catégorielles
    st.subheader('Encodage des variables catégorielles')
    
    le = LabelEncoder()
    encoded_df = df.copy()
    cat_cols = ['Profil', 'Sexe', 'Diplome']
    
    for col in cat_cols:
        encoded_df[f'{col}_encoded'] = le.fit_transform(df[col])
        st.write(f'Encodage pour {col}:')
        for i, label in enumerate(le.classes_):
            st.write(f'{label}: {i}')
    
    # Sélection de la variable cible
    st.subheader('Sélection de la variable cible')
    target_var = st.selectbox('Choisissez la variable cible', ['SAT', 'INT'])
    
    # Distribution de la variable cible
    st.subheader(f'Distribution de {target_var}')
    fig = plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=target_var, kde=True)
    st.pyplot(fig)
    
    # Variables explicatives
    st.subheader('Variables explicatives retenues')
    predictors = ['ADT', 'FOR', 'RE', 'PE', 'EE', 'FC', 'SI']
    if target_var == 'SAT':
        predictors.append('INT')
    elif target_var == 'INT':
        predictors.append('SAT')
    
    st.write('Variables explicatives:', ', '.join(predictors))

# Page 5: Régression linéaire
elif page == 'Régression linéaire':
    st.title('📈 Régression linéaire multiple')
    
    # Préparation des données
    target_var = st.selectbox('Choisissez la variable cible', ['SAT', 'INT'])
    predictors = ['ADT', 'FOR', 'RE', 'PE', 'EE', 'FC', 'SI']
    if target_var == 'SAT':
        predictors.append('INT')
    elif target_var == 'INT':
        predictors.append('SAT')
    
    X = df[predictors]
    y = df[target_var]
    
    # Split des données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Régression
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Résultats
    st.subheader('Résultats de la régression')
    
    # R²
    r2 = model.score(X_test, y_test)
    st.write(f'R² (coefficient de détermination): {r2:.3f}')
    
    # Coefficients
    coef_df = pd.DataFrame({'Variable': predictors, 'Coefficient': model.coef_})
    st.write('Coefficients:')
    st.dataframe(coef_df)
    
    # VIF (Multicolinéarité)
    st.subheader('Analyse de la multicolinéarité (VIF)')
    vif_data = pd.DataFrame()
    vif_data['Variable'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    st.dataframe(vif_data)
    
    # Analyse des résidus
    st.subheader('Analyse des résidus')
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Distribution des résidus
    sns.histplot(residuals, kde=True, ax=ax1)
    ax1.set_title('Distribution des résidus')
    
    # Résidus vs Valeurs prédites
    ax2.scatter(y_pred, residuals)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Valeurs prédites')
    ax2.set_ylabel('Résidus')
    ax2.set_title('Résidus vs Valeurs prédites')
    
    st.pyplot(fig)

# Page 6: Visualisation finale
elif page == 'Visualisation finale':
    st.title('🎯 Visualisation finale')
    
    target_var = st.selectbox('Choisissez la variable cible', ['SAT', 'INT'])
    
    # Prédictions vs Réelles
    predictors = ['ADT', 'FOR', 'RE', 'PE', 'EE', 'FC', 'SI']
    if target_var == 'SAT':
        predictors.append('INT')
    elif target_var == 'INT':
        predictors.append('SAT')
    
    X = df[predictors]
    y = df[target_var]
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Graphique des valeurs prédites vs réelles
    st.subheader('Valeurs prédites vs réelles')
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Valeurs réelles')
    plt.ylabel('Valeurs prédites')
    st.pyplot(fig)
    
    # Groupes à faible/forte satisfaction
    st.subheader(f'Analyse des groupes à faible/forte {target_var}')
    threshold_low = df[target_var].quantile(0.25)
    threshold_high = df[target_var].quantile(0.75)
    
    low_group = df[df[target_var] <= threshold_low]
    high_group = df[df[target_var] >= threshold_high]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f'Profil des répondants avec {target_var} faible:')
        st.write(low_group['Profil'].value_counts())
        
    with col2:
        st.write(f'Profil des répondants avec {target_var} élevé:')
        st.write(high_group['Profil'].value_counts())
    
    # Résumé automatique des insights
    st.subheader('Résumé des insights')
    
    # Corrélations principales
    corr_with_target = df[predictors + [target_var]].corr()[target_var].sort_values(ascending=False)
    
    st.write(f'Principaux facteurs influençant {target_var}:')
    for var, corr in corr_with_target[1:4].items():
        st.write(f'- {var}: corrélation de {corr:.3f}')
    
    # Différences entre groupes
    st.write(f'\nDifférences significatives entre les groupes:')
    for cat_var in ['Profil', 'Sexe', 'Diplome']:
        f_stat, p_val = stats.f_oneway(*[group for _, group in df.groupby(cat_var)[target_var]])
        if p_val < 0.05:
            st.write(f'- Différences significatives selon {cat_var} (p-value = {p_val:.4f})')
