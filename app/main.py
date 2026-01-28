"""
Application Streamlit - Cinéma Creuse
Version complète avec toutes les fonctionnalités + Recommandations
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from pathlib import Path
from datetime import datetime
from streamlit_folium import st_folium

# Imports depuis utils.py
from utils import (
    PALETTE_CREUSE, CINEMAS, VILLES_CREUSE, ACTIVITES_ANNEXES,
    get_project_root, enrich_movie_with_tmdb, format_genre, translate_genres,
    safe_get, check_password, create_map, create_styled_barplot,
    get_now_playing_france, match_now_playing_with_imdb,
    assign_films_to_cinemas, calculate_cinema_distance,
    get_movie_details_from_tmdb, get_films_affiche_enrichis,
    assign_films_to_cinemas_enrichis, find_movies_with_correction,
    display_youtube_video, get_trailers_from_films, check_title_columns,
    UserManager, init_paul_profile_if_needed
)

# ==========================================
# CONFIGURATION STREAMLIT & GESTIONNAIRE UTILISATEUR
# ==========================================
# Initialise UserManager() depuis utils.py pour gérer les profils utilisateurs
# (films aimés/non aimés, historique, préférences de genres)
user_manager = UserManager()

# Pré-remplissage du profil 'Paul' avec 30 films via init_paul_profile_if_needed()
# uniquement si le profil est vide (évite duplication au rechargement)
init_paul_profile_if_needed(user_manager)

st.set_page_config(
    page_title="Votre cinéma en Creuse",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#F8F9FA'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# ==========================================
# CHEMINS PROJET & FONCTIONS DE CHARGEMENT
# ==========================================
# get_project_root() depuis utils.py détecte la racine du projet
PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"


@st.cache_data
def load_excel_data():
    """
    Charge l'ensemble des feuilles Excel du fichier Cinemas_existants_creuse.xlsx
    
    Returns:
        dict: Dictionnaire contenant 11 DataFrames (démographie, prix, confiserie, etc.)
              ou None si erreur de chargement
    
    Note: Utilise pd.read_excel() avec sheet_name pour charger plusieurs feuilles
          Le cache Streamlit évite de recharger à chaque interaction
    """
    excel_path = DATA_DIR / "processed" / 'Cinemas_existants_creuse.xlsx'
    
    if not excel_path.exists():
        return None
    
    try:
        data = {
            'cine_csp_g': pd.read_excel(excel_path, sheet_name='Cine_CSP_Global'),
            'cine_age_g': pd.read_excel(excel_path, sheet_name='Cine_Age_Global'),
            'candies_c': pd.read_excel(excel_path, sheet_name='Confiseries'),
            'movies_type_g': pd.read_excel(excel_path, sheet_name='movies_type_shares'),
            'prog_g': pd.read_excel(excel_path, sheet_name='programmation'),
            'mensual_price': pd.read_excel(excel_path, sheet_name='prix_mensuel'),
            'streaming_price': pd.read_excel(excel_path, sheet_name='prix_streaming'),
            'cine_c': pd.read_excel(excel_path, sheet_name='Cinemas'),
            'pop_c': pd.read_excel(excel_path, sheet_name='Population_creuse'),
            'kids_c': pd.read_excel(excel_path, sheet_name='Enfants_creuse'),
            'dip_c': pd.read_excel(excel_path, sheet_name='Diplome_creuse')
        }
        
        data['streaming_price'].columns = data['streaming_price'].columns.str.strip()
        data['mensual_price'].columns = data['mensual_price'].columns.str.strip()
        
        return data
    except Exception as e:
        st.error(f"Erreur Excel : {e}")
        return None


@st.cache_data
def load_imdb_data():
    """
    Charge et prétraite le dataset IMDb depuis imdb_complet_avec_cast.parquet
    
    Pipeline de traitement :
    1. Lecture Parquet (optimisé pour colonnes larges avec cast)
    2. Renommage de colonnes pour compatibilité (primaryTitle→titre, etc.)
    3. Conversions numériques avec pd.to_numeric(..., errors='coerce')
    4. Transformation genres (string→list via split(','))
    5. Filtres qualité : note>0, votes≥100, durée≥60
    6. Création display_title via get_display_title() pour affichage optimisé
    
    Returns:
        pd.DataFrame: Dataset nettoyé prêt pour KNN et affichage UI
                      ou None si erreur de chargement
    """
    imdb_path = DATA_DIR / 'PARQUETS' / 'imdb_complet_avec_cast.parquet'
    
    if not imdb_path.exists():
        st.error(f"❌ Fichier non trouvé : {imdb_path}")
        return None
    
    try:
        df = pd.read_parquet(imdb_path)
        
        # ==========================================
        # MAPPING DE COLONNES POUR COMPATIBILITÉ UI
        # Renomme primaryTitle→titre, averageRating→note, etc.
        # Vérifie existence avant pour éviter KeyError sur datasets variés
        # ==========================================
        
        column_mapping = {
            'primaryTitle': 'titre',
            'averageRating': 'note',
            'runtimeMinutes': 'durée',
            'numVotes': 'votes'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]
        
        # ==========================================
        # CONVERSIONS NUMÉRIQUES & TRANSFORMATION GENRES
        # pd.to_numeric(..., errors='coerce') convertit invalides→NaN, puis fillna()
        # genres string "Action,Drama" → list ["Action", "Drama"] via split+strip
        # ==========================================
        
        # Conversions numériques avec gestion erreurs (coerce→NaN)
        if 'note' in df.columns:
            df['note'] = pd.to_numeric(df['note'], errors='coerce').fillna(0)
        if 'votes' in df.columns:
            df['votes'] = pd.to_numeric(df['votes'], errors='coerce').fillna(0)
        if 'durée' in df.columns:
            df['durée'] = pd.to_numeric(df['durée'], errors='coerce').fillna(90)
        
        # Genres en liste
        if 'genres' in df.columns:
            df['genre'] = df['genres'].fillna('').apply(
                lambda x: [g.strip() for g in x.split(',')] if isinstance(x, str) and x else []
            )
        
        # ==========================================
        # FILTRES QUALITÉ (PRÉ-SÉLECTION CATALOGUE)
        # Critères minimums : note>0, ≥100 votes, durée≥60min
        # Réduit bruit (films non notés, courts-métrages, contenu marginal)
        # ==========================================
        
        df = df[
            (df.get('note', 0) > 0) &
            (df.get('votes', 0) >= 100) &
            (df.get('durée', 0) >= 60)
        ].copy()
        
        # ==========================================
        # COLONNE display_title POUR PERFORMANCE UI
        # get_display_title() depuis utils.py génère "Titre FR (Année)" ou fallback EN
        # Pré-calcul (1 fois) évite .apply() répété dans boucles d'affichage
        # ==========================================
        
        from utils import get_display_title
        df['display_title'] = df.apply(
            lambda row: get_display_title(row, prefer_french=True, include_year=False),
            axis=1
        )
        
        df = df.reset_index(drop=True)
        
        # Stats de chargement
        st.sidebar.info(f"📊 {len(df):,} films IMDB")
        
        return df
        
    except Exception as e:
        st.error(f"Erreur IMDb : {e}")
        import traceback
        st.error(traceback.format_exc())
        return None


# Charger les données
data = load_excel_data()

with st.spinner("📥 Chargement du catalogue..."):
    df_movies = load_imdb_data()

if df_movies is None:
    st.error("❌ Impossible de charger les films")
    st.stop()

if data is None:
    st.warning("⚠️ Données Excel non disponibles")
    data = {}


# ==========================================
# SYSTÈME DE RECOMMANDATION KNN SIMPLIFIÉ 
# ==========================================
# Utilise uniquement pandas + sklearn de base
# Pas de classes custom, pas de ColumnTransformer complexe
#
# Architecture :
# 1. build_knn_simple() : pandas + StandardScaler + NearestNeighbors (cached)
# 2. get_recommendations_knn() : Trouve les N films similaires
# 3. get_recommendations() : Wrapper avec gestion d'erreurs
# ==========================================

@st.cache_resource
@st.cache_resource(show_spinner="🔄 Construction du modèle KNN...")
def build_knn_simple(df: pd.DataFrame):
    """
    Construit un modèle KNN PROPRE avec ColumnTransformer et Pipeline
    
    Architecture sklearn professionnelle :
    1. Preprocessing : Transformer listes (genres, acteurs, réalisateurs) en colonnes 0/1
    2. ColumnTransformer : Séparer colonnes binaires vs numériques
    3. Pipeline : preprocessor + NearestNeighbors
    
    Args:
        df: DataFrame avec colonnes [genre, acteurs, realisateurs, startYear, durée]
    
    Returns:
        dict: {
            'df_features': DataFrame avec toutes les colonnes préparées,
            'pipeline': Pipeline sklearn complet,
            'preprocessor': ColumnTransformer,
            'binary_cols': Liste des colonnes binaires,
            'numeric_cols': Liste des colonnes numériques
        }
    """
    from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.neighbors import NearestNeighbors
    from collections import Counter
    
    st.sidebar.info("🔄 Étape 1/3 : Préparation des features...")
    
    # ==========================================
    # ÉTAPE 1 : PRÉPARER LES FEATURES
    # ==========================================
    
    # 1.1 GENRES (MultiLabelBinarizer)
    mlb_genres = MultiLabelBinarizer()
    X_genres = mlb_genres.fit_transform(df['genre'])
    df_genres = pd.DataFrame(
        X_genres,
        columns=[f'genre_{g}' for g in mlb_genres.classes_],
        index=df.index
    )
    
    st.sidebar.success(f"✅ Genres : {len(mlb_genres.classes_)} colonnes")
    
    # 1.2 RÉALISATEURS (Top 50)
    df_directors = None
    director_col = None
    
    for col_name in ['realisateurs', 'directors', 'director']:
        if col_name in df.columns:
            director_col = col_name
            break
    
    if director_col:
        directors_list = []
        for directors in df[director_col]:
            if isinstance(directors, (list, tuple, np.ndarray)) and len(directors) > 0:
                directors_list.append(directors[0])
            else:
                directors_list.append('')
        
        director_counts = Counter(directors_list)
        top_directors = [d for d, _ in director_counts.most_common(50) if d != '']
        
        director_data = {}
        for director in top_directors:
            col_name = f'director_{director.replace(" ", "_")[:30]}'
            director_data[col_name] = [
                1 if isinstance(d, (list, tuple, np.ndarray)) and len(d) > 0 and d[0] == director else 0
                for d in df[director_col]
            ]
        
        df_directors = pd.DataFrame(director_data, index=df.index)
        st.sidebar.success(f"✅ Réalisateurs : {len(top_directors)} colonnes")
    else:
        st.sidebar.warning("⚠️ Pas de colonne réalisateur")
    
    # 1.3 ACTEURS (Top 100)
    df_actors = None
    actor_col = None
    
    for col_name in ['acteurs', 'actors', 'cast']:
        if col_name in df.columns:
            actor_col = col_name
            break
    
    if actor_col:
        all_actors = []
        for actors in df[actor_col]:
            if isinstance(actors, (list, tuple, np.ndarray)) and len(actors) > 0:
                all_actors.extend(actors[:5])
        
        actor_counts = Counter(all_actors)
        top_actors = [a for a, _ in actor_counts.most_common(100)]
        
        actor_data = {}
        for actor in top_actors:
            col_name = f'actor_{actor.replace(" ", "_")[:30]}'
            actor_data[col_name] = [
                1 if isinstance(a, (list, tuple, np.ndarray)) and any(act == actor for act in a[:5]) else 0
                for a in df[actor_col]
            ]
        
        df_actors = pd.DataFrame(actor_data, index=df.index)
        st.sidebar.success(f"✅ Acteurs : {len(top_actors)} colonnes")
    else:
        st.sidebar.warning("⚠️ Pas de colonne acteurs")
    
    # 1.4 FEATURES NUMÉRIQUES
    numeric_cols = ['startYear', 'durée']
    df_numeric = df[numeric_cols].copy()
    df_numeric = df_numeric.fillna(df_numeric.median())
    
    # 1.5 COMBINER
    dfs_to_concat = [df_genres, df_numeric]
    if df_directors is not None:
        dfs_to_concat.append(df_directors)
    if df_actors is not None:
        dfs_to_concat.append(df_actors)
    
    df_features = pd.concat(dfs_to_concat, axis=1)
    
    st.sidebar.info("🔄 Étape 2/3 : Construction du Pipeline...")
    
    # ==========================================
    # ÉTAPE 2 : COLUMNSTRANSFORMER + PIPELINE
    # ==========================================
    
    # Identifier colonnes binaires vs numériques
    binary_cols = df_features.loc[:, df_features.nunique() == 2].columns.tolist()
    numeric_cols_final = df_features.drop(binary_cols, axis=1).columns.tolist()
    
    # ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('binary', 'passthrough', binary_cols),
            ('numeric', StandardScaler(), numeric_cols_final)
        ],
        remainder='drop'
    )
    
    # Pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('knn', NearestNeighbors(metric='cosine', algorithm='brute'))
    ])
    
    st.sidebar.info("🔄 Étape 3/3 : Entraînement...")
    
    # Fit
    pipeline.fit(df_features)
    
    # Afficher récapitulatif
    st.sidebar.divider()
    st.sidebar.success(f"✅ Modèle KNN entraîné !")
    st.sidebar.info(f"📊 **{df_features.shape[1]} features totales**")
    st.sidebar.caption(f"  • Genres : {len(df_genres.columns)}")
    st.sidebar.caption(f"  • Numériques : {len(numeric_cols)}")
    if df_directors is not None:
        st.sidebar.caption(f"  • Réalisateurs : {len(df_directors.columns)}")
    if df_actors is not None:
        st.sidebar.caption(f"  • Acteurs : {len(df_actors.columns)}")
    st.sidebar.divider()
    
    return {
        'df_features': df_features,
        'pipeline': pipeline,
        'preprocessor': preprocessor,
        'binary_cols': binary_cols,
        'numeric_cols': numeric_cols_final
    }


def get_recommendations_knn(df: pd.DataFrame, movie_index: int, n: int = 10, min_quality: bool = True):
    """
    Trouve les N films les plus similaires avec Pipeline sklearn
    
    Args:
        df: DataFrame original avec tous les films
        movie_index: Position du film dans le DataFrame (iloc)
        n: Nombre de recommandations à retourner
        min_quality: Si True, filtre les films avec note > 0
    
    Returns:
        pd.DataFrame: Les N films les plus similaires
    """
    # Construire le modèle
    engine = build_knn_simple(df)
    df_features = engine['df_features']
    pipeline = engine['pipeline']
    knn = pipeline.named_steps['knn']
    
    # Transformer les features
    X_transformed = pipeline.named_steps['preprocessor'].transform(df_features)
    
    # Chercher plus de voisins si filtrage activé
    search_neighbors = (n * 3) + 1 if min_quality else n + 1
    
    # KNN
    distances, indices = knn.kneighbors(
        [X_transformed[movie_index]], 
        n_neighbors=search_neighbors
    )
    
    # Retirer le film lui-même
    neighbor_indices = indices[0][1:]
    neighbor_distances = distances[0][1:]
    
    # Récupérer les films
    recommendations = df.iloc[neighbor_indices].copy()
    
    # Ajouter similarité (1 - distance cosine)
    recommendations['similarite'] = 1 - neighbor_distances
    
    # Filtrage qualité optionnel
    if min_quality:
        recommendations = recommendations[recommendations.get('note', 0) > 0]
    
    # Retourner seulement N films
    return recommendations.head(n)


def get_recommendations(df: pd.DataFrame, movie_index: int, n: int = 10):
    """
    Wrapper simple pour gérer les erreurs
    
    Args:
        df: DataFrame avec tous les films
        movie_index: Position du film source
        n: Nombre de recommandations
    
    Returns:
        tuple: (DataFrame des films recommandés, nom de la méthode)
    
    Note: Renvoie DataFrame vide si KNN échoue
    """
    try:
        reco = get_recommendations_knn(df, movie_index, n)
        return reco, "KNN (cosine)"
    except Exception:
        return df.iloc[[]], "KNN (indisponible)"

# ==========================================
# SIDEBAR : NAVIGATION & FILTRES DYNAMIQUES
# ==========================================
# st.sidebar.radio() génère menu de navigation entre 7 pages
# Filtres (genres, note, durée) s'affichent uniquement sur page "🏠 Accueil"
# via condition if page == "🏠 Accueil"

st.sidebar.title("🎬 Navigation")

page = st.sidebar.radio(
    "Choisir une page",
    ["🏠 Accueil", "🎬 Films à l'affiche", "❤️ Mes Films Favoris", "💡 Recommandations", "🗺️ Cinémas Creuse", "🎭 Activités Annexes", "📊 Espace B2B"]
)

st.sidebar.markdown("---")

# Affichage filtres conditionnels (uniquement page Accueil)
# Extraction genres uniques depuis colonne 'genre' (list) via set.update()
# Filtrage DataFrame avec .apply(lambda) pour vérifier intersection genres
if page == "🏠 Accueil":
    st.sidebar.title("🎯 Filtres")
    
    all_genres = set()
    for genres in df_movies['genre']:
        if isinstance(genres, list):
            all_genres.update(genres)
    all_genres = sorted([g for g in all_genres if g])
    
    selected_genres = st.sidebar.multiselect("Genres", options=all_genres, default=[])
    min_rating = st.sidebar.slider("Note minimum", 0.0, 10.0, 6.0, 0.5)
    max_runtime = st.sidebar.slider("Durée max (min)", 60, 240, 180, 10)
    
    df_filtered = df_movies.copy()
    
    if selected_genres:
        df_filtered = df_filtered[
            df_filtered['genre'].apply(
                lambda x: any(g in x for g in selected_genres) if isinstance(x, list) else False
            )
        ]
    
    df_filtered = df_filtered[df_filtered['note'] >= min_rating]
    df_filtered = df_filtered[df_filtered['durée'] <= max_runtime]
else:
    df_filtered = df_movies.copy()

st.sidebar.markdown("---")

# ==========================================
# SYSTÈME AUTHENTIFICATION UTILISATEUR (SIDEBAR)
# ==========================================
# Gère connexion/déconnexion via st.session_state['authenticated']
# - Mode connecté : affiche nom utilisateur + bouton déconnexion
# - Mode invité : affiche formulaire connexion (username/password)
# Authentification via check_password() depuis utils.py
# ==========================================

st.sidebar.subheader("🔐 Connexion")

# Vérification état connexion depuis session Streamlit
if st.session_state.get('authenticated', False):
    # ==========================================
    # UTILISATEUR CONNECTÉ : affichage profil + logout
    # ==========================================
    username = st.session_state.get('authenticated_user', 'Utilisateur')
    
    st.sidebar.success(f"👤 **{username}**")
    st.sidebar.caption("Profil personnalisé actif")
    
    # Bouton déconnexion : reset session_state + rerun interface
    if st.sidebar.button("🚪 Se déconnecter", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.authenticated_user = None
        st.success("Déconnexion réussie")
        st.rerun()

else:
    # ==========================================
    # MODE INVITÉ : formulaire connexion
    # st.sidebar.form évite rerun à chaque saisie clavier
    # Validation via check_password(username, password) depuis utils.py
    # ==========================================
    st.sidebar.info("👤 Mode **Invité**")
    
    with st.sidebar.form("sidebar_login_form"):
        st.caption("Connectez-vous pour un profil personnalisé")
        
        username = st.text_input("Identifiant", key="sidebar_username")
        password = st.text_input("Mot de passe", type="password", key="sidebar_password")
        
        col1, col2 = st.columns(2)
        with col1:
            submit = st.form_submit_button("✅ Connexion", use_container_width=True)
        with col2:
            cancel = st.form_submit_button("❌ Annuler", use_container_width=True)
        
        if submit:
            # Vérifier les identifiants
            from utils import ADMIN_CREDENTIALS
            
            if username in ADMIN_CREDENTIALS and ADMIN_CREDENTIALS[username] == password:
                st.session_state.authenticated = True
                st.session_state.authenticated_user = username
                st.success(f"✅ Bienvenue {username} !")
                st.rerun()
            else:
                st.error("❌ Identifiant ou mot de passe incorrect")
    
    st.sidebar.caption("💡 **Identifiants** : paul / WCS26")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**📊 {len(df_movies):,} films**")
st.sidebar.markdown("**📅 Année : 2026**")
st.sidebar.markdown("**🎓 Wild Code School**")


# ==========================================
# PAGE : ACCUEIL (DOCUMENTATION TECHNIQUE)
# ==========================================
# Affiche architecture projet avec 5 sections :
# 1. Présentation (info box)
# 2. Architecture données (IMDb vs TMDb)
# 3. Workflow (diagramme matplotlib avec FancyBboxPatch)
# 4. Statistiques (métriques + graphiques seaborn)
# 5. Stack technique (colonnes technologies)
# ==========================================

if page == "🏠 Accueil":
    st.title("🎬 Cinéma Creuse - Documentation Technique")
    st.markdown("### Architecture et méthodologie du projet")
    
    # ==========================================
    # SECTION 1 : PRÉSENTATION PROJET
    # Encadré st.info() avec contexte structurel/conjoncturel
    # ==========================================
    
    st.info("""
    **Bienvenue sur la plateforme Cinéma Creuse !**
    
    Ce projet combine des **données structurelles** historiques (IMDb) avec des **données conjoncturelles** 
    en temps réel (TMDb) pour offrir une expérience de recommandation de films complète et moderne.
    """)
    
    st.markdown("---")
    
    # ==========================================
    # SECTION 2 : ARCHITECTURE DONNÉES (DUAL SOURCE)
    # Colonnes comparant IMDb (statique) vs TMDb (temps réel)
    # - IMDb : load_imdb_data() → parquet local → KNN
    # - TMDb : get_films_affiche_enrichis() → API → page Films à l'affiche
    # ==========================================
    
    st.header("📊 Architecture des données")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🗄️ Données structurelles : IMDb")
        st.success("""
        **Base statique historique**
        
        📁 **Source** : IMDb public datasets
        
        📊 **Contenu** :
        - 10M+ de titres catalogués (films, séries, etc.)
        - 5M+ titres retenus :
            - Distribution : France
            - Type : Film
            - Années 1990-2026
        - 22K- films disposant des informations nécessaires (acteurs, réalisateur, votes, titre français, etc.)
        
        🎯 **Usage** :
        - Base de recommandations
        - Système KNN et similarité
        - Matching avec TMDb
        - Analyses statistiques
        """)
        
        st.metric("Films IMDb", f"{len(df_movies):,}")
        st.metric("Note moyenne", f"{df_movies['note'].mean():.1f}/10")
    
    with col2:
        st.subheader("🌐 Données conjoncturelles : TMDb")
        st.info("""
        **API temps réel**
        
        🔗 **Source** : The Movie Database API
        
        📊 **Contenu** :
        - Films à l'affiche (now_playing)
        - Films à venir (upcoming)
        - Affiches officielles HD et trailers
        - Synopsis français
        - Casting et équipe complets
        
        🎯 **Usage** :
        - Page Films à l'affiche
        - Enrichissement visuels
        - Page Cinémas
        - Mode dégradé (cache)
        """)
        
        try:
            films = get_films_affiche_enrichis()
            st.metric("Films TMDb", len(films))
        except:
            st.metric("Films TMDb", "18 (cache)")
    
    
    # ==========================================
    # SECTION 4 : STATISTIQUES CATALOGUE (MÉTRIQUES + GRAPHIQUES)
    # - st.metric() pour KPIs (total, moyenne, récents, nb genres)
    # - matplotlib.pyplot : courbe temporelle + histogramme + barh horizontal
    # - Palette PALETTE_CREUSE depuis utils.py pour cohérence visuelle
    # ==========================================
    
    st.markdown("---")
    st.header("📈 Statistiques de la base")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Films totaux", f"{len(df_movies):,}")
    with col2:
        st.metric("Note moyenne", f"{df_movies['note'].mean():.2f}/10")
    with col3:
        films_2020 = len(df_movies[df_movies['startYear'] >= 2020])
        st.metric("Films ≥ 2020", f"{films_2020:,}")
    with col4:
        genres = set()
        for g in df_movies['genre']:
            if isinstance(g, list):
                genres.update(g)
        st.metric("Genres", len(genres))
    
    # Graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 Films par année")
        df_years = df_movies[df_movies['startYear'] >= 1970]
        year_counts = df_years.groupby('startYear').size()
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(year_counts.index, year_counts.values, color=PALETTE_CREUSE['principal'], lw=2)
        ax.fill_between(year_counts.index, year_counts.values, alpha=0.3, color=PALETTE_CREUSE['secondaire'])
        ax.set_xlabel('Année')
        ax.set_ylabel('Nombre de films')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("⭐ Distribution des notes")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(df_movies['note'], bins=25, color=PALETTE_CREUSE['principal'], 
                edgecolor='black', alpha=0.7)
        ax.axvline(df_movies['note'].mean(), color='red', linestyle='--', lw=2)
        ax.set_xlabel('Note /10')
        ax.set_ylabel('Nombre de films')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close() 
    
    # Calcul top genres via comptage dict manuel (évite dépendance collections.Counter)
    # Tri par nombre d'occurrences décroissant → top 10
    st.subheader("🎭 Top 10 des genres")
    genre_counts = {}
    for genres in df_movies['genre']:
        if isinstance(genres, list):
            for g in genres:
                genre_counts[g] = genre_counts.get(g, 0) + 1
    
    top = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh([g[0] for g in top], [g[1] for g in top], 
                    color=PALETTE_CREUSE['gradient'])
    ax.set_xlabel('Nombre de films')
    ax.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, (_, val)) in enumerate(zip(bars, top)):
        ax.text(val + 50, i, f'{val:,}', va='center', fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # ==========================================
    # SECTION 5 : STACK TECHNIQUE (3 COLONNES)
    # Présentation technologies utilisées par catégorie :
    # - Data & ML : Pandas, NumPy, Scikit-learn (NearestNeighbors)
    # - Web & API : Streamlit, requests, TMDb, Folium (cartes interactives)
    # - Storage : Parquet (colonnes optimisées), cache local, mode dégradé TMDb
    # ==========================================
    
    st.markdown("---")
    st.header("🛠️ Technologies utilisées")
    
    col1, col2, col3 = st.columns(3) 
    
    with col1:
        st.subheader("📊 Data & ML")
        st.markdown("""
        - Pandas, NumPy
        - Scikit-learn (KNN)
        - Matplotlib, Seaborn
        """)
    
    with col2:
        st.subheader("🌐 Web & API")
        st.markdown("""
        - Streamlit
        - Requests, TMDb API
        - Folium (cartes)
        """)
    
    with col3:
        st.subheader("💾 Storage")
        st.markdown("""
        - Parquet (IMDb)
        - Cache statique
        - Mode dégradé
        """)
    
    # Footer
    st.markdown("---")
    st.success("""
    🎓 **Wild Code School 2026** - Projet Data Analysis
    
    👥 Équipe : Paul, Hamidou, Lynda | 🎯 Cinémas de la Creuse
    """)

elif page == "🎬 Films à l'affiche":
    st.title("🎬 Films à l'affiche en France")
    st.markdown("Découvrez tous les films en salles maintenant et ceux qui arrivent bientôt !") 
    
    # ==========================================
    # EXPANDER PÉDAGOGIQUE : EXPLICATION API TMDb
    # ==========================================
    with st.expander("Comprendre le système dactualisation des films API TMDb", expanded=False, icon="🔄"):
        col1, col2, col3 = st.columns([1, 8, 1])

        with col2:
            st.caption("🎓 Découvrez en 4 étapes comment notre système récupère les films actuellement en salles et affiche leurs informations en temps réel.")

            st.markdown("**🌐 Compte TMDb → 📡 Requête API → 💾 Cache 24h → 🎬 Affichage dynamique**")
            st.divider()

            # ÉTAPE 1
            st.subheader("🔑 Étape 1 — Créer un compte développeur TMDb")
            st.markdown(
                "**TMDb (The Movie Database)** est une base de données collaborative de films avec une API gratuite.\n\n"
                "**🎯 Pourquoi TMDb plutôt qu'IMDb ?**\n"
                "- IMDb n'a **pas d'API publique gratuite** 💰\n"
                "- TMDb propose une API avec **3000 requêtes gratuites par jour** ✅\n"
                "- Les données sont **mises à jour en temps réel** par la communauté\n\n"
                "**📝 Comment créer un compte ?**\n"
                "```python\n"
                "# 1. Aller sur https://www.themoviedb.org/\n"
                "# 2. Créer un compte gratuit\n"
                "# 3. Dans Paramètres → API → Demander une clé API\n"
                "# 4. Remplir le formulaire (usage éducatif/personnel)\n"
                "# 5. Récupérer votre API_KEY (une longue chaîne de caractères)\n"
                "```\n\n"
            )

            # ÉTAPE 2
            st.subheader("📡 Étape 2 — Effectuer des requêtes API")
            st.markdown(
                "Une **API (Application Programming Interface)** permet à deux programmes de communiquer.\n\n"
                "**🎬 Exemple concret : Récupérer les films à l'affiche**\n"
                "```python\n"
                "import requests\n\n"
                "# URL de l'API TMDb pour les films en salle (now_playing)\n"
                "url = 'https://api.themoviedb.org/3/movie/now_playing'\n\n"
                "# Paramètres de la requête\n"
                "params = {\n"
                "    'api_key': 'VOTRE_CLE_API',      # Votre clé secrète\n"
                "    'language': 'fr-FR',              # Langue française\n"
                "    'region': 'FR'                    # Films en France\n"
                "}\n\n"
                "# Envoyer la requête GET\n"
                "response = requests.get(url, params=params)\n\n"
                "# Récupérer les données au format JSON\n"
                "films = response.json()['results']  # Liste de films\n"
                "```\n\n"
                "**🔍 Que contient la réponse ?**\n"
                "```python\n"
                "# Pour chaque film, on reçoit :\n"
                "film = {\n"
                "    'id': 12345,                      # ID unique TMDb\n"
                "    'title': 'Inception',             # Titre français\n"
                "    'original_title': 'Inception',    # Titre original\n"
                "    'release_date': '2010-07-16',     # Date de sortie\n"
                "    'vote_average': 8.8,              # Note moyenne /10\n"
                "    'overview': 'Dom Cobb est...',    # Synopsis\n"
                "    'poster_path': '/abc123.jpg',     # Chemin de l'affiche\n"
                "    'genre_ids': [28, 878, 53]        # IDs des genres\n"
                "}\n"
                "```\n\n"
                "💡 **Astuce** : TMDb a une excellente **documentation interactive** sur https://developers.themoviedb.org/ "
                "où on peut tester les requêtes directement dans le navigateur !"
            )

            # ÉTAPE 3
            st.subheader("💾 Étape 3 — Système de cache (24 heures)")
            st.markdown(
                "**Problème** : Si on appelle l'API à chaque visite, on va vite atteindre la limite de 3000 requêtes/jour.\n\n"
                "**Solution** : Mettre en **cache** les résultats pendant 24 heures.\n\n"
                "**🔄 Comment ça marche ?**\n"
                "```python\n"
                "import streamlit as st\n"
                "from datetime import datetime, timedelta\n\n"
                "@st.cache_data(ttl=86400)  # ttl = 86400 secondes = 24 heures\n"
                "def get_films_affiche_enrichis():\n"
                "    '''Récupère les films à l'affiche avec cache de 24h'''\n"
                "    \n"
                "    # 1. Streamlit vérifie si les données sont déjà en cache\n"
                "    # 2. Si oui ET que < 24h → retourne le cache (pas de requête API)\n"
                "    # 3. Si non OU que > 24h → appelle l'API et met à jour le cache\n"
                "    \n"
                "    films = requests.get(url, params=params).json()\n"
                "    return films\n"
                "```\n\n"
                "**✅ Avantages du cache**\n"
                "- ⚡ **Rapidité** : Pas d'attente réseau (affichage instantané)\n"
                "- 💰 **Économie de requêtes** : 100 utilisateurs = 1 seule requête API\n"
                "- 🛡️ **Mode dégradé** : Si l'API est en panne, on affiche quand même le cache\n\n"
                "**⚠️ Inconvénient**\n"
                "Les données peuvent avoir jusqu'à 24h de retard. Pour les films en salle, c'est acceptable !\n\n"
                "💡 **Fallback** : Si l'API ne répond pas ET qu'il n'y a pas de cache, on charge un fichier JSON statique "
                "avec ~18 films populaires (mode dégradé)."
            )

            # ÉTAPE 4
            st.subheader("🎬 Étape 4 — Enrichissement et affichage")
            st.markdown(
                "Les données TMDb sont **brutes**. On doit les enrichir pour l'affichage.\n\n"
                "**🔧 Traitement dans `get_films_affiche_enrichis()`**\n"
                "```python\n"
                "def get_films_affiche_enrichis():\n"
                "    # 1. Récupérer films from TMDb API\n"
                "    films_raw = get_now_playing_france()\n"
                "    \n"
                "    # 2. Pour chaque film, enrichir les données\n"
                "    films_enrichis = []\n"
                "    for film in films_raw:\n"
                "        enrichi = {\n"
                "            'tmdb_id': film['id'],\n"
                "            'titre': film['title'],\n"
                "            'note': film['vote_average'],\n"
                "            \n"
                "            # Construire URL complète de l'affiche\n"
                "            'poster_url': f\"https://image.tmdb.org/t/p/w500{film['poster_path']}\",\n"
                "            \n"
                "            # Récupérer détails supplémentaires (réalisateur, acteurs)\n"
                "            'realisateur': get_movie_details_from_tmdb(film['id'])['director'],\n"
                "            'acteurs': get_movie_details_from_tmdb(film['id'])['cast'][:5],\n"
                "            \n"
                "            # Convertir genre_ids en noms\n"
                "            'genres': [GENRE_MAP[gid] for gid in film['genre_ids']]\n"
                "        }\n"
                "        films_enrichis.append(enrichi)\n"
                "    \n"
                "    return films_enrichis\n"
                "```\n\n"
                "**🎨 Affichage dans Streamlit**\n"
                "```python\n"
                "for film in films_enrichis:\n"
                "    col1, col2 = st.columns([1, 3])\n"
                "    \n"
                "    with col1:\n"
                "        st.image(film['poster_url'])  # Affiche l'affiche\n"
                "    \n"
                "    with col2:\n"
                "        st.markdown(f\"**{film['titre']}**\")\n"
                "        st.write(f\"⭐ {film['note']}/10\")\n"
                "        st.write(f\"🎬 {film['realisateur']}\")\n"
                "        st.write(f\"🎭 {', '.join(film['genres'])}\")\n"
                "```"
            )

            # BONUS
            st.markdown("---")
            st.markdown("**💡 Fonctionnalités avancées de notre système**")
            st.markdown(
                "**🎥 Extraction des trailers YouTube**\n"
                "```python\n"
                "def get_trailers_from_films(films, max_trailers=5):\n"
                "    '''Récupère les trailers YouTube depuis l'API TMDb'''\n"
                "    trailers = {}\n"
                "    \n"
                "    for film in films[:max_trailers]:  # Limiter pour rate limit\n"
                "        # Appel endpoint /movie/{id}/videos\n"
                "        videos = requests.get(f\"{BASE_URL}/movie/{film['tmdb_id']}/videos\").json()\n"
                "        \n"
                "        # Chercher la bande-annonce officielle YouTube\n"
                "        for video in videos['results']:\n"
                "            if video['type'] == 'Trailer' and video['site'] == 'YouTube':\n"
                "                trailers[film['tmdb_id']] = {\n"
                "                    'video_id': video['key'],      # ID YouTube\n"
                "                    'titre': film['titre'],\n"
                "                    'realisateur': film['realisateur']\n"
                "                }\n"
                "                break\n"
                "    \n"
                "    return trailers\n"
                "```\n\n"
                "**🔄 Séparation par statut (en salle vs à venir)**\n"
                "```python\n"
                "def separer_films_par_statut(films):\n"
                "    '''Sépare selon release_date vs date actuelle'''\n"
                "    today = datetime.now().date()\n"
                "    \n"
                "    films_en_salles = []\n"
                "    films_bientot = []\n"
                "    \n"
                "    for film in films:\n"
                "        release = datetime.strptime(film['date_sortie'], '%Y-%m-%d').date()\n"
                "        \n"
                "        if release <= today:\n"
                "            films_en_salles.append(film)   # Déjà sorti\n"
                "        else:\n"
                "            films_bientot.append(film)      # Pas encore sorti\n"
                "    \n"
                "    return films_en_salles, films_bientot\n"
                "```\n\n"
                "**🎯 Matching avec notre base IMDb**\n"
                "Pour certains films, on peut croiser les données TMDb avec notre base IMDb locale "
                "via le titre + année pour récupérer des infos supplémentaires (casting complet, notes détaillées)."
            )

        
    # ==========================================
    # RÉCUPÉRATION FILMS TMDb (API + CACHE FALLBACK)
    # get_films_affiche_enrichis() depuis utils.py :
    # - Appel TMDb API (now_playing + upcoming)
    # - Enrichissement via get_movie_details_from_tmdb()
    # - Fallback cache si API indisponible
    # ==========================================
    with st.spinner("🎬 Récupération des films..."):
        films_affiche = get_films_affiche_enrichis()
    
    if not films_affiche:
        st.warning("⚠️ Impossible de récupérer les films à l'affiche pour le moment.")
        st.stop()
    
    # ==========================================
    # EXTRACTION TRAILERS YOUTUBE
    # get_trailers_from_films() depuis utils.py :
    # - Cherche video_id YouTube pour chaque film
    # - Limite à max_trailers pour performance (TMDb rate limit)
    # - Retourne dict {tmdb_id: {video_id, titre, realisateur, ...}}
    # ==========================================
    with st.spinner("🎥 Recherche des trailers disponibles..."):
        trailers_disponibles = get_trailers_from_films(films_affiche, max_trailers=5)
    
    # Affichage trailer du film le plus populaire (si disponible)
    if trailers_disponibles:
        st.markdown("### 🎥 Bande-annonce du moment")
        
        # Tri par popularité (field TMDb) → premier=plus populaire
        films_avec_trailers = [
            (key, info) for key, info in trailers_disponibles.items()
        ]
        
        films_avec_trailers.sort(
            key=lambda x: x[1]['film_data'].get('popularite', 0),
            reverse=True
        )
        
        # Affichage via display_youtube_video() (iframe embed personnalisé)
        if films_avec_trailers:
            selected_key, trailer_info = films_avec_trailers[0]
            
            display_youtube_video(
                video_id=trailer_info['video_id'],
                title=trailer_info['titre'],
                director=trailer_info['realisateur'],
                max_width=900
            )
            
            # Métriques film (note, année, durée)
            film_data = trailer_info['film_data']
            col1, col2, col3 = st.columns(3)
            with col1:
                if film_data.get('note'):
                    st.metric("Note", f"⭐ {film_data['note']}/10")
            with col2:
                if film_data.get('annee'):
                    st.metric("Année", film_data['annee'])
            with col3:
                if film_data.get('duree'):
                    st.metric("Durée", f"{film_data['duree']} min")
        
        st.markdown("---")
    
    # ==========================================
    # SÉPARATION FILMS PAR STATUT (RELEASE_DATE)
    # separer_films_par_statut() depuis utils.py compare release_date vs datetime.now()
    # Retourne (films_en_salles, films_bientot) selon statut TMDb
    # ==========================================
    from utils import separer_films_par_statut
    films_en_salles, films_bientot = separer_films_par_statut(films_affiche)
    
    st.success(f"✅ {len(films_en_salles)} films en salles • 🔜 {len(films_bientot)} films à venir")
    
    # st.tabs() sépare UX (évite scroll infini)
    tab1, tab2 = st.tabs([
        f"🎬 Déjà en salles ({len(films_en_salles)})",
        f"🔜 Bientôt disponibles ({len(films_bientot)})"
    ])
    
    # ==========================================
    # TAB 1 : FILMS EN SALLES (FILTRES + PAGINATION + GRID)
    # - Filtres sidebar : genres (multiselect), note (slider)
    # - Tri : popularité, note, titre (A-Z/Z-A)
    # - Pagination manuelle via st.session_state.page_num_salles
    # - Affichage grille 4 colonnes avec posters + expander détails
    # ==========================================
    
    with tab1:
        if not films_en_salles:
            st.info("Aucun film actuellement en salles.")
        else:
            # Filtres dans la sidebar
            st.sidebar.title("🎯 Filtres (Films en salles)")
            
            # Genres
            all_genres_salles = set()
            for film in films_en_salles:
                if film.get('genres'):
                    all_genres_salles.update(film['genres'])
            all_genres_salles = sorted(list(all_genres_salles))
            
            selected_genres_salles = st.sidebar.multiselect(
                "Genres", 
                options=all_genres_salles, 
                default=[],
                key="genres_salles"
            )
            
            min_rating_salles = st.sidebar.slider(
                "Note minimum", 
                0.0, 10.0, 0.0, 0.5,
                key="rating_salles"
            )
            
            # Filtrer
            films_salles_filtres = films_en_salles.copy()
            
            if selected_genres_salles:
                films_salles_filtres = [
                    film for film in films_salles_filtres
                    if film.get('genres') and any(g in film['genres'] for g in selected_genres_salles)
                ]
            
            if min_rating_salles > 0:
                films_salles_filtres = [
                    film for film in films_salles_filtres
                    if film.get('note', 0) >= min_rating_salles
                ]
            
            # Options d'affichage
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"**{len(films_salles_filtres)} films** en salles")
            
            with col2:
                sort_by_salles = st.selectbox(
                    "Trier par",
                    ["Popularité", "Note (desc)", "Note (asc)", "Titre (A-Z)", "Titre (Z-A)"],
                    key="sort_salles"
                )
            
            with col3:
                per_page_salles = st.selectbox("Par page", [12, 24, 48], index=1, key="per_page_salles")
            
            # Tri
            if sort_by_salles == "Popularité":
                films_salles_sorted = sorted(films_salles_filtres, key=lambda x: x.get('popularite', 0), reverse=True)
            elif sort_by_salles == "Note (desc)":
                films_salles_sorted = sorted(films_salles_filtres, key=lambda x: x.get('note', 0), reverse=True)
            elif sort_by_salles == "Note (asc)":
                films_salles_sorted = sorted(films_salles_filtres, key=lambda x: x.get('note', 0))
            elif sort_by_salles == "Titre (A-Z)":
                films_salles_sorted = sorted(films_salles_filtres, key=lambda x: x.get('titre', ''))
            else:  # Z-A
                films_salles_sorted = sorted(films_salles_filtres, key=lambda x: x.get('titre', ''), reverse=True)
            
            # Pagination
            total_pages_salles = (len(films_salles_sorted) - 1) // per_page_salles + 1 if films_salles_sorted else 1
            
            if 'page_num_salles' not in st.session_state:
                st.session_state.page_num_salles = 1
            
            if st.session_state.page_num_salles > total_pages_salles:
                st.session_state.page_num_salles = 1
            
            col_prev, col_page, col_next = st.columns([1, 2, 1])
            
            with col_prev:
                if st.button("⬅️ Précédent", key="prev_salles") and st.session_state.page_num_salles > 1:
                    st.session_state.page_num_salles -= 1
                    st.rerun()
            
            with col_page:
                st.markdown(f"**Page {st.session_state.page_num_salles} / {total_pages_salles}**")
            
            with col_next:
                if st.button("Suivant ➡️", key="next_salles") and st.session_state.page_num_salles < total_pages_salles:
                    st.session_state.page_num_salles += 1
                    st.rerun()
            
            st.markdown("---")
            
            # Affichage
            if films_salles_sorted:
                start_idx = (st.session_state.page_num_salles - 1) * per_page_salles
                end_idx = start_idx + per_page_salles
                page_films = films_salles_sorted[start_idx:end_idx]
                
                cols = st.columns(4)
                
                for idx, film in enumerate(page_films):
                    with cols[idx % 4]:
                        # Affichage du film (même code qu'avant)
                        st.image(film['poster_url'], use_container_width=True)
                        
                        titre = film.get('titre', 'Sans titre')
                        st.markdown(f"**{titre[:35]}{'...' if len(titre) > 35 else ''}**")
                        
                        note = film.get('note', 0)
                        nb_votes = film.get('nb_votes', 0)
                        st.markdown(f"⭐ {note:.1f}/10")
                        if nb_votes > 0:
                            st.caption(f"📊 {nb_votes:,} votes")
                        
                        if film.get('annee'):
                            st.caption(f"📅 {film['annee']}")
                        
                        if film.get('duree'):
                            st.caption(f"⏱️ {film['duree']} min")
                        
                        genres = film.get('genres', [])
                        if genres:
                            genres_traduits = translate_genres(genres[:2])
                            st.caption(f"🎭 {', '.join(genres_traduits)}")
                        
                        with st.expander("📄 Voir les détails"):
                            st.markdown("**📝 Synopsis**")
                            st.markdown(film['synopsis'])
                            
                            st.markdown("---")
                            
                            if film.get('realisateur') and film['realisateur'] != 'Inconnu':
                                st.markdown(f"**🎬 Réalisateur** : {film['realisateur']}")
                            
                            if film.get('acteurs'):
                                st.markdown(f"**👥 Acteurs principaux** : {', '.join(film['acteurs'][:5])}")
                            
                            if genres:
                                st.markdown(f"**🎭 Genres** : {', '.join(genres)}")
                            
                            if film.get('date_sortie'):
                                st.markdown(f"**📅 Sortie** : {film['date_sortie']}")
                            
                            if film.get('langue_originale'):
                                st.markdown(f"**🌍 Langue** : {film['langue_originale'].upper()}")
                            
                            if film.get('titre_original') and film['titre_original'] != film['titre']:
                                st.caption(f"*Titre original : {film['titre_original']}*")
            else:
                st.info("Aucun film ne correspond à vos critères.")
    
    # ==========================================
    # TAB 2 : FILMS À VENIR
    # ==========================================
    
    with tab2:
        if not films_bientot:
            st.info("Aucun film à venir prochainement.")
        else:
            st.markdown("### 🔜 Films qui sortiront bientôt en France")
            
            # Tri par date de sortie (plus proche d'abord)
            films_bientot_sorted = sorted(films_bientot, key=lambda x: x.get('date_sortie', ''))
            
            # Affichage en grille
            cols = st.columns(4)
            
            for idx, film in enumerate(films_bientot_sorted):
                with cols[idx % 4]:
                    # Badge "À venir"
                    st.markdown("🔜 **BIENTÔT**")
                    
                    st.image(film['poster_url'], use_container_width=True)
                    
                    titre = film.get('titre', 'Sans titre')
                    st.markdown(f"**{titre[:35]}{'...' if len(titre) > 35 else ''}**")
                    
                    # Date de sortie mise en avant
                    if film.get('date_sortie'):
                        from datetime import datetime
                        try:
                            date_sortie = datetime.strptime(film['date_sortie'], '%Y-%m-%d')
                            st.markdown(f"📅 **{date_sortie.strftime('%d/%m/%Y')}**")
                        except:
                            st.markdown(f"📅 **{film['date_sortie']}**")
                    
                    note = film.get('note', 0)
                    if note > 0:
                        st.markdown(f"⭐ {note:.1f}/10")
                    
                    if film.get('duree'):
                        st.caption(f"⏱️ {film['duree']} min")
                    
                    genres = film.get('genres', [])
                    if genres:
                        genres_traduits = translate_genres(genres[:2])
                        st.caption(f"🎭 {', '.join(genres_traduits)}")
                    
                    with st.expander("📄 Voir les détails"):
                        st.markdown("**📝 Synopsis**")
                        st.markdown(film['synopsis'])
                        
                        st.markdown("---")
                        
                        if film.get('realisateur') and film['realisateur'] != 'Inconnu':
                            st.markdown(f"**🎬 Réalisateur** : {film['realisateur']}")
                        
                        if film.get('acteurs') and len(film['acteurs']) > 0:
                            st.markdown(f"**👥 Acteurs principaux** : {', '.join(film['acteurs'][:5])}")
                        
                        if genres:
                            st.markdown(f"**🎭 Genres** : {', '.join(genres)}")
                        
                        if film.get('date_sortie'):
                            st.markdown(f"**📅 Sortie prévue** : {film['date_sortie']}")
                        
                        if film.get('langue_originale'):
                            st.markdown(f"**🌍 Langue** : {film['langue_originale'].upper()}")
                        
                        if film.get('titre_original') and film['titre_original'] != film['titre']:
                            st.caption(f"*Titre original : {film['titre_original']}*")



# ==========================================
# PAGE : RECOMMANDATIONS (2 MODES)
# ==========================================
# Mode 1 (Tab1) : Recommandations personnalisées basées sur profil utilisateur
#   - get_personalized_recommendations() depuis utils.py
#   - Analyse films aimés/genres préférés via UserManager
#   - Score pondéré (similarité KNN + préférences genres)
# Mode 2 (Tab2) : Recherche par titre/acteur
#   - find_movies_with_correction() pour recherche fuzzy
#   - get_recommendations() pour KNN sur film sélectionné
# ==========================================

elif page == "💡 Recommandations":
    st.title("🎬 Système de Recommandation de Films")
    
    # ==========================================
    # EXPANDER PÉDAGOGIQUE : EXPLICATION KNN EN 6 ÉTAPES
    # Documentation complète méthodologie (optionnel, collapsed par défaut)
    # Couvre : séparation données, preprocessing, cosine, KNN, limites
    # ==========================================
    with st.expander("Comprendre le système de recommandation KNN", expanded=False, icon="📚"):
        col1, col2, col3 = st.columns([1, 8, 1])

        with col2:
            st.caption("Découvrez comment notre système trouve les films similaires")

            st.markdown("**Chunks → Nettoyage → Préparation → Preprocessing → Pipeline → KNN → Similarité**")
            st.divider()

            # ÉTAPE 0
            st.subheader("Étape 0 — Récupération des données par chunks", divider= True)
            st.markdown(
                "Avant toute chose, on récupère les données d'IMDb de manière optimisée :\n\n"
                "- On ne conserve que les titres distribués en France, grâce à <u>title.akas.tsv.gz</u> via une analyse par chunks.\n\n"
                "```python\n"
                "tconst_france= {}\n"
                "chunks = []\n"
                "chunk_size = 500_000\n\n"
                "for chunk in pd.read_csv('title.akas.tsv.gz', sep='\\t', chunksize=chunk_size):\n"
                "   films_fr = chunk[chunk['region'] == 'FR']['titleId'].unique()\n"
                "   tconst_france.update(films_fr)\n\n" 
                "```\n\n"       
                "- **Filtrage de <u>title.basics.tsv.gz</u> par chunks (éviter de charger 10M+ lignes)**, en conservant que les films distribués en France (liste précédente) et disposant d'un vote.\n\n"
                "```python\n"
                "chunks = []\n"
                "chunk_size = 500_000\n\n"
                "for chunk in pd.read_csv('title.basics.tsv', sep='\\t', chunksize=chunk_size):\n"
                "    filtered = chunk[(chunk['titleType'] == 'movie') & (chunk['averageRating'] > 0)] & (chunk['tconst'].isin(tconst_france))\n"
                "    chunks.append(filtered)\n\n"
                "df_movies = pd.concat(chunks, ignore_index=True)\n"
                "```\n\n"
                "- **Jointure avec acteurs/réalisateurs** issus de la table <u>title.principals.tsv.gz</u>\n"
                "```python\n"
                "acteurs = df_cast[df_cast['category'].isin(['actor', 'actress'])].groupby('tconst')['primaryName'].apply(list)\n"
                "realisateurs = df_cast[df_cast['category'] == 'director'].groupby('tconst')['primaryName'].apply(list)\n"
                "df_movies = df_movies.merge(acteurs, on='tconst').merge(realisateurs, on='tconst')\n"
                "```",
                unsafe_allow_html=True
            )

            # ÉTAPE 1
            st.subheader("Étape 1 — Nettoyage des données de notre DataFrame principal : df_movies", divider= True)
            st.markdown(
                "```python\n"
                "df_movies = df_movies[df_movies['startYear'].notna()]\n"
                "df_movies = df_movies[(df_movies['runtimeMinutes'] >= 40) & (df_movies['runtimeMinutes'] <= 300)]\n"
                "df_movies['genre'] = df_movies['genres'].str.split(',')\n"
                "```"
            )

            # ÉTAPE 2
            st.subheader("Étape 2 — Préparation des données", divider= True)
            st.markdown(
                "On sélectionne **TOUTES** les colonnes pour calculer la similarité :\n\n"
                "**5 types de features :**\n"
                "- **Genres** : ['Action', 'Sci-Fi'] → genre_Action=1, genre_Sci-Fi=1\n"
                "- **Année** : 2010 → sera standardisé\n"
                "- **Durée** : 148 min → sera standardisé\n"
                "- **Réalisateurs** : ['Christopher Nolan']\n"
                "- **Acteurs** : ['Leo DiCaprio', 'Tom Hardy']\n\n"
            )

            # ÉTAPE 3
            st.subheader("Étape 3 — Preprocessing", divider= True)
            st.markdown(
                "**Problème : Les listes ne sont pas utilisables directement**\n\n"
                "```python\n"
                "df['genre'] = [['Action', 'Sci-Fi'], ...]  # ❌ KNN ne comprend pas\n"
                "```\n\n"
                "**Pourquoi pas OneHotEncoder ?**\n"
                "```python\n"
                "X = [['Action', 'Sci-Fi'], ['Drama']]\n"
                "OneHotEncoder().fit(X)  # ❌ TypeError: unhashable type: 'list'\n"
                "```\n\n"
                "**Solution : MultiLabelBinarizer**\n"
                "```python\n"
                "from sklearn.preprocessing import MultiLabelBinarizer\n\n"
                "mlb = MultiLabelBinarizer()\n"
                "X_genres = mlb.fit_transform(df['genre'])\n"
                "# [[1 0 0 1 0]  ← Action=1, Sci-Fi=1\n"
                "#  [0 0 1 0 0]] ← Drama=1\n"
                "```\n\n"
                "MultiLabelBinarizer > OneHotEncoder car conçu pour multi-label !"
            )
            # ÉTAPE 4
            st.subheader("Étape 4 — Pipeline sklearn", divider= True)
            st.markdown(
                "```python\n"
                "from sklearn.compose import ColumnTransformer\n"
                "from sklearn.pipeline import Pipeline\n\n"
                "# Séparer binaires vs numériques\n"
                "preprocessor = ColumnTransformer([\n"
                "    ('binary', 'passthrough', binary_cols),  # Genres, acteurs, réalisateurs\n"
                "    ('numeric', StandardScaler(), numeric_cols)  # Année, durée\n"
                "])\n\n"
                "pipeline = Pipeline([\n"
                "    ('preprocessor', preprocessor),\n"
                "    ('knn', NearestNeighbors(metric='cosine'))\n"
                "])\n\n"
                "pipeline.fit(df_features)\n"
                "```"
            )
            st.image('https://i.ytimg.com/vi/kccT0FVK6OY/maxresdefault.jpg')
            # ÉTAPE 5
            st.subheader("Étape 5 — Entraîner et utiliser le KNN", divider= True)
            st.markdown(
                "```python\n"
                "X_transformed = pipeline.named_steps['preprocessor'].transform(df_features)\n"
                "knn = pipeline.named_steps['knn']\n\n"
                "distances, indices = knn.kneighbors([X_transformed[42]], n_neighbors=11)\n"
                "neighbor_indices = indices[0][1:]  # Retirer le film lui-même\n"
                "```\n\n"
                "Distance cosine = angle entre vecteurs → Angle petit = Films similaires"
            )

            # ÉTAPE 6
            st.subheader("Étape 6 — Calcul de la similarité", divider= True)
            st.markdown(
                "**Pourquoi calculer la similarité ?**\n\n"
                "KNN retourne des distances, on veut des similarités pour l'utilisateur :\n"
                "```python\n"
                "similarite = 1 - distance\n\n"
                "# distance = 0.12 → similarite = 88% ✅\n"
                "# distance = 0.75 → similarite = 25% ❌\n\n"
                "recommendations['similarite'] = 1 - neighbor_distances\n"
                "```\n\n"
                "**Relation avec KNN :**\n"
                "KNN trouve voisins → Calcule distances → 1-distance = similarité → Affichage"
            )

            # ÉTAPE 7
            st.subheader("Étape 7 — Récupérer et afficher", divider= True)
            st.markdown(
                "```python\n"
                "def get_recommendations_knn(df, movie_index, n=10):\n"
                "    engine = build_knn_simple(df)\n"
                "    pipeline = engine['pipeline']\n"
                "    X_transformed = pipeline.named_steps['preprocessor'].transform(engine['df_features'])\n"
                "    knn = pipeline.named_steps['knn']\n"
                "    \n"
                "    distances, indices = knn.kneighbors([X_transformed[movie_index]], n_neighbors=n+1)\n"
                "    neighbor_indices = indices[0][1:]\n"
                "    \n"
                "    recommendations = df.iloc[neighbor_indices].copy()\n"
                "    recommendations['similarite'] = 1 - distances[0][1:]\n"
                "    return recommendations.head(n)\n"
                "```"
            )

            # ÉTAPE 8
            st.subheader("Étape 8 — Applications du KNN : 3 cas d'usage différents", divider= True)
            st.markdown(
                "Le MÊME modèle KNN est utilisé de 3 façons différentes dans l'application :\n\n"
                "---\n\n"
                "### 1️⃣ Recherche par film (Films similaires)\n\n"
                "**Cas d'usage** : L'utilisateur sélectionne UN film, on recommande des films similaires\n\n"
                "**Fonctionnement** :\n"
                "```python\n"
                "# Utilisateur choisit 'Inception'\n"
                "film_index = df[df['titre'] == 'Inception'].index[0]  # Position : 42\n\n"
                "# KNN cherche les voisins de CE film précis\n"
                "distances, indices = knn.kneighbors(\n"
                "    [X_transformed[film_index]],  # Vecteur d'Inception\n"
                "    n_neighbors=11\n"
                ")\n\n"
                "# Résultat : Films similaires à Inception\n"
                "# → Interstellar, The Dark Knight, The Prestige (tous Nolan)\n"
                "```\n\n"
                "**Logique** :\n"
                "- Point de départ : UN film connu\n"
                "- Recherche : Quels autres films ont un vecteur similaire ?\n"
                "- Base de comparaison : Les 177 features du film (genres, année, durée, réalisateur, acteurs)\n\n"
                "---\n\n"
                "### 2️⃣ Recherche par acteur\n\n"
                "**Cas d'usage** : L'utilisateur cherche des films avec UN acteur spécifique\n\n"
                "**Fonctionnement** :\n"
                "```python\n"
                "# Utilisateur cherche 'Tom Hanks'\n"
                "films_tom_hanks = df[\n"
                "    df['acteurs'].apply(lambda x: 'Tom Hanks' in x if isinstance(x, list) else False)\n"
                "]\n\n"
                "# Prendre UN film de référence (ex : le plus populaire)\n"
                "film_reference = films_tom_hanks.sort_values('note', ascending=False).iloc[0]\n"
                "film_index = film_reference.name\n\n"
                "# KNN cherche les voisins de CE film\n"
                "distances, indices = knn.kneighbors([X_transformed[film_index]], n_neighbors=50)\n\n"
                "# Filtrer pour garder SEULEMENT les films avec Tom Hanks\n"
                "recommendations = df.iloc[indices[0]]\n"
                "recommendations_filtered = recommendations[\n"
                "    recommendations['acteurs'].apply(lambda x: 'Tom Hanks' in x)\n"
                "]\n"
                "```\n\n"
                "**Logique** :\n"
                "- Point de départ : UN film de Tom Hanks (le plus populaire)\n"
                "- Recherche : Autres films similaires\n"
                "- Filtrage APRÈS : Ne garder que ceux avec Tom Hanks\n"
                "- Résultat : Films Tom Hanks similaires au film de référence\n\n"
                "**Pourquoi cette approche ?**\n"
                "- On ne peut pas créer un vecteur fictif 'Tom Hanks'\n"
                "- On utilise un VRAI film comme point de départ\n"
                "- Le KNN trouve des films similaires (même époque, mêmes genres...)\n"
                "- Le filtrage garantit que Tom Hanks est présent\n\n"
                "**Exemple** :\n"
                "- Film de référence : *Forrest Gump* (Drama, Romance • 1994 • Tom Hanks)\n"
                "- KNN trouve : Cast Away, The Green Mile, Saving Private Ryan\n"
                "- Tous ont Tom Hanks + genres/époque similaires\n\n"
                "---\n\n"
                "### 3️⃣ Films favoris (Recommandations personnalisées)\n\n"
                "**Cas d'usage** : L'utilisateur a aimé PLUSIEURS films, on recommande des films qu'il pourrait aimer\n\n"
                "**Fonctionnement** :\n"
                "```python\n"
                "# Utilisateur a aimé 5 films\n"
                "films_favoris = ['Inception', 'The Dark Knight', 'Interstellar', 'The Matrix', 'Blade Runner 2049']\n\n"
                "# Récupérer les indices\n"
                "indices_favoris = df[df['titre'].isin(films_favoris)].index\n\n"
                "# MÉTHODE : CENTROÏDE (vecteur moyen)\n"
                "vecteurs_favoris = X_transformed[indices_favoris]\n"
                "vecteur_moyen = vecteurs_favoris.mean(axis=0)  # Moyenne des 5 vecteurs\n\n"
                "# KNN cherche les voisins du vecteur moyen\n"
                "distances, indices = knn.kneighbors(\n"
                "    [vecteur_moyen],  # Point fictif = moyenne des goûts\n"
                "    n_neighbors=50\n"
                ")\n"
                "```\n\n"
                "**Logique** :\n"
                "- Point de départ : Vecteur MOYEN des films aimés\n"
                "- Représente le 'profil de goût' de l'utilisateur\n"
                "- KNN trouve des films proches de ce profil moyen\n\n"
                "**Exemple vecteur moyen** :\n"
                "```python\n"
                "# Inception :    [1, 0, 1, 1, ..., 1, 0]  (Action, Sci-Fi, Nolan)\n"
                "# Matrix :       [1, 0, 1, 0, ..., 0, 1]  (Action, Sci-Fi)\n"
                "# Dark Knight :  [1, 1, 0, 1, ..., 1, 0]  (Action, Crime, Nolan)\n"
                "#                 ↓  ↓  ↓  ↓       ↓  ↓\n"
                "# Moyenne :      [1, 0.3, 0.7, 0.7, ..., 0.7, 0.3]\n"
                "#                ↑ Action probable (100%)\n"
                "#                   ↑ Un peu Crime (30%)\n"
                "#                      ↑ Beaucoup Sci-Fi (70%)\n"
                "```\n\n"
                "Le vecteur moyen crée un 'film fictif' qui représente les goûts !\n\n"
                "---\n\n"
                "### 📊 Comparaison des 3 méthodes\n\n"
            )
            
            # Tableau comparatif
            comparison_data = {
                "Critère": [
                    "Point de départ",
                    "Nombre de vecteurs",
                    "Calcul KNN",
                    "Filtrage après",
                    "Personnalisation",
                    "Use case"
                ],
                "Par film": [
                    "1 film connu",
                    "1 vecteur réel",
                    "kneighbors([vecteur_film])",
                    "Aucun",
                    "❌ Non",
                    "Explorer similaires"
                ],
                "Par acteur": [
                    "1 film de l'acteur",
                    "1 vecteur réel",
                    "kneighbors([vecteur_film])",
                    "✅ Garde acteur",
                    "❌ Non",
                    "Découvrir filmographie"
                ],
                "Films favoris": [
                    "N films aimés",
                    "N vecteurs → moyenne",
                    "kneighbors([vecteur_moyen])",
                    "✅ Retire favoris",
                    "✅✅ Oui",
                    "Recommandations perso"
                ]
            }
            
            st.table(comparison_data)
            
            st.markdown(
                "\n**Points clés** :\n"
                "1. **Films similaires** : Simple, direct, 1 film → voisins\n"
                "2. **Par acteur** : 1 film de référence + filtrage pour garantir l'acteur\n"
                "3. **Favoris** : Agrégation de goûts → vecteur moyen = profil utilisateur\n\n"
                "---\n\n"
                "### 💡 Pourquoi 3 approches pour 1 modèle ?\n\n"
                "**Le KNN est flexible** :\n"
                "- Peut chercher voisins d'UN point (film)\n"
                "- Peut chercher voisins d'un point MOYEN (profil)\n"
                "- Peut être combiné avec filtrage\n\n"
                "**Même modèle, 3 questions différentes** :\n"
                "- 'Quels films ressemblent à Inception ?' → Par film\n"
                "- 'Quels films Tom Hanks similaires ?' → Par acteur\n"
                "- 'Qu'est-ce que je vais aimer ?' → Favoris\n\n"
                "**Avantage** : 1 seul modèle à entraîner, 3 fonctionnalités !\n"
            )

            # EXEMPLE
            st.markdown("---")
            st.markdown("**Exemple : Inception**")
            st.markdown(
                "Film : Inception (Action, Sci-Fi, Thriller • 2010 • 148 min)\n\n"
                "Résultats :\n"
                "1. Interstellar (88%) → Même réalisateur (Nolan)\n"
                "2. The Dark Knight (85%) → Même réalisateur (Nolan)\n"
                "3. The Prestige (82%) → Même réalisateur (Nolan)"
            )

            # FORCES ET LIMITES
            st.markdown("---")
            st.subheader("Forces et limites")
            st.markdown("**Forces** : Rapide, simple, explicable, flexible")
            st.markdown("**Limites** : Cold start, popularité, contexte, subjectivité")


    st.markdown("### Découvrez des films qui correspondent à vos goûts")
    
    # Extraction utilisateur actuel depuis st.session_state (géré par système auth)
    current_user = st.session_state.get('authenticated_user', 'invite')
    
    # Affichage contexte utilisateur
    if current_user != 'invite':
        st.info(f"👤 Profil de **{current_user}**")
    else:
        st.info("👤 Mode Invité - Connectez-vous pour sauvegarder votre profil")
    
    st.markdown("---")
    
    # Récupération préférences utilisateur via UserManager (utils.py)
    # liked_films/disliked_films : listes de tconst pour filtrage et scoring
    liked_films = user_manager.get_liked_films(current_user)
    disliked_films = user_manager.get_disliked_films(current_user)
    
    # ==========================================
    # TABS : 2 MODES DE RECOMMANDATION DISTINCTS
    # Tab1 : Recommandations personnalisées (profil utilisateur)
    # Tab2 : Recherche manuelle (titre/acteur) + KNN sur sélection
    # ==========================================
    
    tab1, tab2 = st.tabs([
        f"🎯 Recommandations Personnalisées ({len(liked_films)} films aimés)",
        "🔍 Recherche par Titre ou Acteur"
    ])
    
    # ==========================================
    # TAB 1 : RECOMMANDATIONS PERSONNALISÉES
    # Workflow :
    # 1. Vérification profil (liked_films non vide)
    # 2. get_personalized_recommendations(df, liked, disliked, top_n)
    #    → Analyse genres préférés + KNN multiple + scoring pondéré
    # 3. Filtrage interactif (sliders score/nombre)
    # 4. Enrichissement TMDb (affiches) + affichage grille
    # ==========================================
    
    with tab1:
        st.markdown("### 🎯 Films recommandés pour vous")
        
        if len(liked_films) == 0:
            st.info("💡 **Aucun film aimé dans votre profil**")
            st.markdown("""
            Pour recevoir des recommandations personnalisées :
            1. Allez sur la page **❤️ Mes Films Favoris**
            2. Recherchez des films que vous avez aimés
            3. Cliquez sur 👍 pour les ajouter
            4. Revenez ici pour voir vos recommandations !
            """)
        
        else:
            st.markdown(f"*Basées sur vos **{len(liked_films)} films aimés** et vos genres préférés*")
            
            # get_personalized_recommendations() depuis utils.py :
            # - Calcule genres préférés (fréquence dans liked_films)
            # - Pour chaque liked, trouve N voisins KNN
            # - Score composite : similarité KNN × poids genre × pénalité disliked
            # - Retourne DataFrame trié par score_recommandation (0-100)
            from utils import get_personalized_recommendations
            
            # Génération recommandations (peut prendre quelques secondes si profil large)
            with st.spinner("🎬 Génération de vos recommandations personnalisées..."):
                recommended_films = get_personalized_recommendations(
                    df_movies, 
                    liked_films, 
                    disliked_films, 
                    top_n=20
                )
            
            if len(recommended_films) > 0:
                st.success(f"✨ **{len(recommended_films)} films recommandés** pour vous !")
                
                # Sliders interactifs pour filtrage temps réel (sans rerun complet)
                col_opt1, col_opt2 = st.columns(2)
                with col_opt1:
                    nb_to_show = st.slider("Nombre de films à afficher", 5, 20, 10, step=5, key="slider_nb_films")
                with col_opt2:
                    min_score = st.slider("Score minimum (%)", 0, 100, 50, step=10, key="slider_score")
                
                # Filtrage DataFrame par score_recommandation (colonne ajoutée par get_personalized_recommendations)
                films_filtered = recommended_films[
                    recommended_films.get('score_recommandation', 0) >= min_score
                ]
                
                st.markdown("---")
                
                if len(films_filtered) == 0:
                    st.warning(f"Aucun film avec un score >= {min_score}%. Réduisez le score minimum.")
                else:
                    # ==========================================
                    # AFFICHAGE GRILLE FILMS RECOMMANDÉS
                    # Pour chaque film :
                    # - enrich_movie_with_tmdb() récupère poster via TMDb ID matching
                    # - Layout 3 colonnes : poster + infos + actions (like/dislike)
                    # - st.progress() pour visualisation score_recommandation
                    # ==========================================
                    for idx, film in films_filtered.head(nb_to_show).iterrows():
                        
                        # Enrichissement TMDb pour affiche (fallback placeholder si échec)
                        from utils import enrich_movie_with_tmdb, get_display_title
                        film_enrichi = enrich_movie_with_tmdb(film)
                        
                        col_poster, col_info, col_actions = st.columns([1, 3, 1])
                        
                        with col_poster:
                            # Afficher l'affiche
                            st.image(film_enrichi['poster_url'], use_container_width=True)
                        
                        with col_info:
                            # Titre français prioritaire
                            titre_display = get_display_title(film, prefer_french=True, include_year=True)
                            note = film.get('note', film.get('averageRating', 0))
                            
                            # Genres (traduits en français)
                            genres = film.get('genre', [])
                            if isinstance(genres, list) and len(genres) > 0:
                                genres_traduits = translate_genres(genres[:3])
                                genres_str = ', '.join(genres_traduits)
                            else:
                                genres_str = str(film.get('genres', ''))
                            
                            score_reco = film.get('score_recommandation', 0)
                            
                            st.markdown(f"**{titre_display}**")
                            st.markdown(f"⭐ {note:.1f}/10 | 🎭 {genres_str}")
                            
                            # Barre de progression du score de recommandation
                            st.progress(score_reco / 100, text=f"Correspondance : {score_reco:.0f}%")
                            
                            # AJOUTER EXPANDER POUR SYNOPSIS
                            with st.expander("📄 Voir le synopsis"):
                                st.markdown("**📝 Synopsis**")
                                synopsis = film_enrichi.get('synopsis', 'Synopsis non disponible.')
                                st.markdown(synopsis)
                                
                                st.markdown("---")
                                
                                # Réalisateur
                                if film_enrichi.get('director') and film_enrichi['director'] != 'Inconnu':
                                    st.markdown(f"**🎬 Réalisateur** : {film_enrichi['director']}")
                                
                                # Acteurs
                                if film_enrichi.get('cast') and len(film_enrichi['cast']) > 0:
                                    st.markdown(f"**👥 Acteurs** : {', '.join(film_enrichi['cast'][:5])}")
                                
                                # Durée
                                if film_enrichi.get('runtime'):
                                    st.markdown(f"**⏱️ Durée** : {film_enrichi['runtime']} min")
                        
                        with col_actions:
                            # Vérifier si déjà vu
                            film_id = film.get('tconst')
                            already_rated = user_manager.is_film_already_rated(current_user, film_id)
                            
                            if already_rated:
                                if already_rated == 'liked':
                                    st.success("✅ Aimé")
                                else:
                                    st.error("❌ Pas aimé")
                            else:
                                # Boutons pour ajouter
                                if st.button("👍", key=f"tab1_reco_like_{film_id}", use_container_width=True):
                                    user_manager.add_film(current_user, film, 'liked')
                                    st.success("Ajouté !")
                                    st.rerun()
                                
                                if st.button("👎", key=f"tab1_reco_dislike_{film_id}", use_container_width=True):
                                    user_manager.add_film(current_user, film, 'disliked')
                                    st.info("Noté")
                                    st.rerun()
                        
                        st.markdown("---")
            else:
                st.warning("Aucune recommandation trouvée. Essayez d'ajouter plus de films aimés !")
    
    # ==========================================
    # TAB 2 : RECHERCHE MANUELLE
    # ==========================================
    
    with tab2:
        st.markdown("### 🔍 Trouvez des films similaires")
        st.markdown("*Cherchez par titre de film ou par nom d'acteur/réalisateur*")
        
        # Options de recherche
        col_type, col_search = st.columns([1, 4])
        
        with col_type:
            search_type = st.selectbox(
                "Type",
                options=['Titre', 'Acteur', 'Tout'],
                help="Chercher par titre de film ou nom d'acteur/réalisateur",
                key="search_type_tab2"
            )
        
        with col_search:
            placeholders = {
                'Titre': "Ex: Les Évadés, Inception...",
                'Acteur': "Ex: Brad Pitt, Marion Cotillard...",
                'Tout': "Ex: Inception, Christopher Nolan..."
            }
            
            search_query = st.text_input(
                "Recherche",
                placeholder=placeholders[search_type],
                label_visibility="collapsed",
                help="Vous pouvez chercher en français ou en anglais !",
                key="search_tab2"
            )
        
        # Options avancées
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            prefer_french = st.checkbox(
                "🇫🇷 Priorité français", 
                value=True, 
                help="Prioriser les titres français",
                key="prefer_french_tab2",
                disabled=(search_type == 'Acteur')
            )
        
        with col2:
            show_poster = st.checkbox(
                "🖼️ Afficher affiches", 
                value=True,
                help="Afficher les affiches de films",
                key="show_poster_tab2"
            )
        
        with col3:
            search_button = st.button("🔍 Rechercher", use_container_width=True, key="search_btn_tab2")
        
        # Résultats de recherche
        if search_query or search_button:
            
            # Convertir type de recherche
            search_type_param = {
                'Titre': 'title',
                'Acteur': 'actor',
                'Tout': 'all'
            }[search_type]
            
            # Recherche combinée
            from utils import find_movies_combined
            matching_movies, search_message = find_movies_combined(
                search_query, 
                df_movies, 
                max_results=15,
                search_type=search_type_param,
                prefer_french=prefer_french
            )
            
            # Afficher le message
            if search_message:
                # Si le message contient "colonnes" ou "dataset", c'est une erreur de configuration
                if "colonnes" in search_message.lower() or "dataset" in search_message.lower():
                    st.error(search_message)
                else:
                    st.info(search_message)
            
            if len(matching_movies) == 0:
                st.warning(f"❌ Aucun résultat pour '{search_query}'")
                
                # Message d'aide différent selon le type de recherche
                if search_type_param == 'actor':
                    st.info(
                        "💡 **Conseils pour la recherche par acteur :**\n\n"
                        "• Essayez avec seulement le **nom de famille** (ex: 'Pitt' au lieu de 'Brad Pitt')\n"
                        "• Essayez des variations : 'DiCaprio' ou 'Di Caprio'\n"
                        "• Vérifiez l'orthographe\n"
                        "• Certains acteurs peuvent ne pas avoir de films dans le dataset filtré"
                    )
                else:
                    st.info("💡 Essayez en français ou en anglais")
            
            else:
                st.success(f"✅ {len(matching_movies)} résultat(s)")
                
                st.markdown("---")
                st.subheader("📋 Résultats de recherche")
                
                for idx, (_, movie) in enumerate(matching_movies.iterrows()):
                    
                    if show_poster:
                        # Avec affiche
                        col_poster, col_info, col_action = st.columns([1, 3, 1])
                        
                        with col_poster:
                            # Enrichir pour l'affiche
                            from utils import enrich_movie_with_tmdb
                            film_enrichi = enrich_movie_with_tmdb(movie)
                            st.image(film_enrichi['poster_url'], use_container_width=True)
                        
                        with col_info:
                            # Affichage français uniquement
                            from utils import get_display_title
                            display_title = get_display_title(movie, prefer_french=True, include_year=True)
                            rating = movie.get('note', movie.get('averageRating', 0))
                            votes = movie.get('votes', movie.get('numVotes', 0))
                            
                            st.markdown(f"**{display_title}**")
                            st.markdown(f"⭐ {rating:.1f}/10")
                            
                            if votes > 0:
                                st.caption(f"🗳️ {votes:,} votes")
                            
                            # Genres (traduits en français)
                            if 'genre' in movie.index and isinstance(movie['genre'], list) and len(movie['genre']) > 0:
                                genres_traduits = translate_genres(movie['genre'][:3])
                                genres_str = " · ".join(genres_traduits)
                                st.caption(f"🎭 {genres_str}")
                            
                            # Acteurs si recherche acteur
                            if search_type_param in ['actor', 'all']:
                                if 'acteurs' in movie.index and hasattr(movie.get('acteurs'), '__iter__') and not isinstance(movie.get('acteurs'), str):
                                    try:
                                        acteurs_list = list(movie['acteurs'])[:3]
                                        actors_str = ", ".join(acteurs_list)
                                        st.caption(f"👥 {actors_str}")
                                    except:
                                        pass
                        
                        with col_action:
                            # Bouton pour voir les similaires
                            show_similar_key = f"show_similar_{idx}"
                            if st.button(f"🎬 Voir similaires", key=f"tab2_reco_{idx}", use_container_width=True):
                                # Toggle : si déjà affiché, cacher, sinon afficher
                                if show_similar_key in st.session_state and st.session_state[show_similar_key]:
                                    st.session_state[show_similar_key] = False
                                else:
                                    st.session_state[show_similar_key] = True
                                st.rerun()
                        
                        # Afficher les films similaires en carrousel si demandé
                        if show_similar_key in st.session_state and st.session_state[show_similar_key]:
                            st.markdown("---")
                            st.caption(f"**🎬 Films similaires à {display_title} :**")
                            
                            try:
                                # Générer les recommandations
                                movie_idx = movie.name
                                if movie_idx in df_movies.index:
                                    reco_df, method = get_recommendations(df_movies, movie_idx, n=6)
                                    
                                    if len(reco_df) > 0:
                                        # Afficher en carrousel (colonnes)
                                        cols = st.columns(6)
                                        for i, (_, reco_movie) in enumerate(reco_df.iterrows()):
                                            with cols[i]:
                                                # Enrichir pour l'affiche
                                                enriched = enrich_movie_with_tmdb(reco_movie)
                                                st.image(enriched['poster_url'], use_container_width=True)
                                                st.caption(enriched['title'][:25] + ('...' if len(enriched['title']) > 25 else ''))
                                                if enriched['rating']:
                                                    st.caption(f"⭐ {enriched['rating']:.1f}")
                                                
                                                # AJOUTER EXPANDER POUR SYNOPSIS
                                                with st.expander("📄 Détails"):
                                                    st.markdown("**📝 Synopsis**")
                                                    st.markdown(enriched.get('synopsis', 'Synopsis non disponible'))
                                                    
                                                    if enriched.get('director') and enriched['director'] != 'Inconnu':
                                                        st.caption(f"🎬 {enriched['director']}")
                                                    
                                                    if enriched.get('runtime'):
                                                        st.caption(f"⏱️ {enriched['runtime']} min")
                                                    
                                                    if enriched.get('genres'):
                                                        st.caption(f"🎭 {', '.join(enriched['genres'][:2])}")
                                    else:
                                        st.caption("Aucune recommandation")
                                else:
                                    st.caption("Film non trouvé")
                            except Exception as e:
                                st.caption(f"Erreur : {str(e)}")
                    
                    else:
                        # Sans affiche (compact)
                        col1, col2 = st.columns([1, 4])
                        
                        with col1:
                            st.markdown(f"**{idx+1}.**")
                        
                        with col2:
                            # Affichage français uniquement
                            from utils import get_display_title
                            display_title = get_display_title(movie, prefer_french=True, include_year=True)
                            rating = movie.get('note', movie.get('averageRating', 0))
                            votes = movie.get('votes', movie.get('numVotes', 0))
                            
                            st.markdown(f"**{display_title}** - ⭐ {rating:.1f}/10")
                            
                            if votes > 0:
                                st.caption(f"🗳️ {votes:,} votes")
                            
                            # Genres (traduits en français)
                            if 'genre' in movie.index and isinstance(movie['genre'], list) and len(movie['genre']) > 0:
                                genres_traduits = translate_genres(movie['genre'][:3])
                                genres_str = " · ".join(genres_traduits)
                                st.caption(f"🎭 {genres_str}")
                            
                            # Bouton pour voir similaires
                            show_similar_key = f"show_similar_{idx}"
                            if st.button(f"🎬 Voir les recommandations", key=f"tab2_reco_{idx}"):
                                if show_similar_key in st.session_state and st.session_state[show_similar_key]:
                                    st.session_state[show_similar_key] = False
                                else:
                                    st.session_state[show_similar_key] = True
                                st.rerun()
                        
                        # Afficher les films similaires si demandé
                        if show_similar_key in st.session_state and st.session_state[show_similar_key]:
                            st.caption(f"**Films similaires à {display_title} :**")
                            try:
                                movie_idx = movie.name
                                if movie_idx in df_movies.index:
                                    reco_df, method = get_recommendations(df_movies, movie_idx, n=6)
                                    
                                    if len(reco_df) > 0:
                                        cols = st.columns(6)
                                        for i, (_, reco_movie) in enumerate(reco_df.iterrows()):
                                            with cols[i]:
                                                enriched = enrich_movie_with_tmdb(reco_movie)
                                                st.image(enriched['poster_url'], use_container_width=True)
                                                st.caption(enriched['title'][:20] + '...' if len(enriched['title']) > 20 else enriched['title'])
                                                if enriched['rating']:
                                                    st.caption(f"⭐ {enriched['rating']:.1f}")
                                                
                                                # AJOUTER EXPANDER POUR SYNOPSIS
                                                with st.expander("📄 Détails"):
                                                    st.markdown("**📝 Synopsis**")
                                                    st.markdown(enriched.get('synopsis', 'Synopsis non disponible'))
                                                    
                                                    if enriched.get('director') and enriched['director'] != 'Inconnu':
                                                        st.caption(f"🎬 {enriched['director']}")
                                                    
                                                    if enriched.get('runtime'):
                                                        st.caption(f"⏱️ {enriched['runtime']} min")
                                                    
                                                    if enriched.get('genres'):
                                                        st.caption(f"🎭 {', '.join(enriched['genres'][:2])}")
                                    else:
                                        st.caption("Aucune recommandation")
                                else:
                                    st.caption("Film non trouvé")
                            except Exception as e:
                                st.caption(f"Erreur : {str(e)}")
                    
                    st.markdown("---")
        
elif page == "❤️ Mes Films Favoris":
    st.title("❤️ Mes Films Favoris")
    
    # ==========================================
    # EXPANDER PÉDAGOGIQUE : EXPLICATION SYSTÈME PROFILS
    # ==========================================
    with st.expander("Comprendre le système de profils utilisateurs", expanded=False, icon="👤"):
        col1, col2, col3 = st.columns([1, 8, 1])

        with col2:
            st.caption("🎓 Découvrez comment le système sauvegarde vos préférences et améliore vos recommandations.")

            st.markdown("**👤 Profil → 💾 Stockage → 👍👎 Likes/Dislikes → 🎯 Recommandations**")
            st.divider()

            # ÉTAPE 1
            st.subheader("👤 Étape 1 — Le système UserManager")
            st.markdown(
                "**UserManager** est une classe Python qui gère tous les profils utilisateurs.\n\n"
                "**🎯 Qu'est-ce qu'un profil ?**\n"
                "```python\n"
                "# Structure d'un profil utilisateur\n"
                "profil = {\n"
                "    'username': 'paul',\n"
                "    'liked_films': ['tt1375666', 'tt0816692', ...],   # Liste des tconst aimés\n"
                "    'disliked_films': ['tt0111161', ...],              # Liste des tconst pas aimés\n"
                "    'favorite_genres': ['Action', 'Sci-Fi', 'Drama']  # Genres préférés (déduits)\n"
                "}\n"
                "```\n\n"
                "**🔧 Classe UserManager**\n"
                "```python\n"
                "class UserManager:\n"
                "    def __init__(self):\n"
                "        self.profiles = {}  # Dict stockant tous les profils\n"
                "    \n"
                "    def add_liked_film(self, username, tconst):\n"
                "        '''Ajoute un film à la liste des films aimés'''\n"
                "        if username not in self.profiles:\n"
                "            self.profiles[username] = {'liked_films': [], 'disliked_films': []}\n"
                "        self.profiles[username]['liked_films'].append(tconst)\n"
                "    \n"
                "    def get_liked_films(self, username):\n"
                "        '''Récupère tous les films aimés d'un utilisateur'''\n"
                "        return self.profiles.get(username, {}).get('liked_films', [])\n"
                "```\n\n"
                "💡 **Où sont stockés les profils ?**\n"
                "Les profils sont stockés en **mémoire RAM** pendant la session. Quand tu fermes l'application, ils disparaissent. "
                "Pour une vraie app en production, on utiliserait une base de données (SQLite, PostgreSQL)."
            )

            # ÉTAPE 2
            st.subheader("💾 Étape 2 — Système de likes/dislikes")
            st.markdown(
                "Chaque fois que tu cliques sur 👍 ou 👎, voici ce qui se passe :\n\n"
                "**🔄 Workflow complet**\n"
                "```python\n"
                "# 1. L'utilisateur clique sur 👍 pour 'Inception'\n"
                "if st.button('👍', key='like_tt1375666'):\n"
                "    \n"
                "    # 2. On récupère le tconst du film\n"
                "    tconst = 'tt1375666'\n"
                "    \n"
                "    # 3. On l'ajoute au profil via UserManager\n"
                "    user_manager.add_liked_film(current_user, tconst)\n"
                "    \n"
                "    # 4. On retire des dislikes si présent (switch)\n"
                "    user_manager.remove_disliked_film(current_user, tconst)\n"
                "    \n"
                "    # 5. Streamlit recharge la page\n"
                "    st.rerun()\n"
                "```\n\n"
                "**🎭 Déduction des genres préférés**\n"
                "```python\n"
                "def calculate_favorite_genres(liked_films, df_movies):\n"
                "    '''Calcule les genres les plus présents dans les films aimés'''\n"
                "    \n"
                "    genre_counts = {}\n"
                "    \n"
                "    for tconst in liked_films:\n"
                "        # Récupérer le film dans le DataFrame\n"
                "        film = df_movies[df_movies['tconst'] == tconst].iloc[0]\n"
                "        \n"
                "        # Compter chaque genre\n"
                "        for genre in film['genre']:  # ['Action', 'Sci-Fi']\n"
                "            genre_counts[genre] = genre_counts.get(genre, 0) + 1\n"
                "    \n"
                "    # Trier par fréquence décroissante\n"
                "    favorite_genres = sorted(genre_counts.items(), \n"
                "                            key=lambda x: x[1], \n"
                "                            reverse=True)[:5]  # Top 5\n"
                "    \n"
                "    return [genre for genre, count in favorite_genres]\n"
                "```\n\n"
                "💡 **Exemple concret**\n"
                "Si tu aimes : *Inception*, *Interstellar*, *The Dark Knight*\n"
                "→ Genres détectés : Action (3), Sci-Fi (2), Thriller (2)\n"
                "→ Tes genres préférés : Action, Sci-Fi, Thriller"
            )

            # ÉTAPE 3
            st.subheader("🎯 Étape 3 — Impact sur les recommandations")
            st.markdown(
                "Ton profil est utilisé dans `get_personalized_recommendations()` :\n\n"
                "**📊 Score de recommandation pondéré**\n"
                "```python\n"
                "def get_personalized_recommendations(df, liked_films, disliked_films, top_n=20):\n"
                "    '''Génère recommandations basées sur profil utilisateur'''\n"
                "    \n"
                "    # 1. Calculer genres préférés depuis liked_films\n"
                "    favorite_genres = calculate_favorite_genres(liked_films, df)\n"
                "    \n"
                "    recommendations = []\n"
                "    \n"
                "    # 2. Pour chaque film aimé, trouver voisins KNN\n"
                "    for liked_tconst in liked_films:\n"
                "        idx = df[df['tconst'] == liked_tconst].index[0]\n"
                "        neighbors = get_recommendations_knn(df, idx, n=10)\n"
                "        \n"
                "        # 3. Pour chaque voisin, calculer score\n"
                "        for _, film in neighbors.iterrows():\n"
                "            \n"
                "            # Score de base (similarité KNN) = 50%\n"
                "            score = 50\n"
                "            \n"
                "            # Bonus si genres correspondent (+30%)\n"
                "            if any(g in favorite_genres for g in film['genre']):\n"
                "                score += 30\n"
                "            \n"
                "            # Bonus si note élevée (+20%)\n"
                "            if film['note'] >= 7.5:\n"
                "                score += 20\n"
                "            \n"
                "            # Pénalité si déjà dans disliked (-100 = exclusion)\n"
                "            if film['tconst'] in disliked_films:\n"
                "                score = 0\n"
                "            \n"
                "            recommendations.append({\n"
                "                'film': film,\n"
                "                'score_recommandation': min(score, 100)  # Plafonné à 100\n"
                "            })\n"
                "    \n"
                "    # 4. Dédupliquer et trier par score\n"
                "    recommendations = sorted(recommendations, \n"
                "                            key=lambda x: x['score_recommandation'], \n"
                "                            reverse=True)[:top_n]\n"
                "    \n"
                "    return recommendations\n"
                "```\n\n"
                "✅ **Résultat**\n"
                "Plus tu likes/dislikes de films, plus le système comprend tes goûts !"
            )

            # RÉCAP
            st.markdown("---")
            st.markdown("**📋 Récapitulatif : Comment tout se connecte**")
            st.markdown(
                "```\n"
                "1. 👤 Tu te connectes (ou mode Invité)\n"
                "   ↓\n"
                "2. 🔍 Tu recherches un film (find_movies_with_correction)\n"
                "   ↓\n"
                "3. 👍 Tu cliques sur J'aime\n"
                "   ├─ UserManager.add_liked_film(user, tconst)\n"
                "   └─ Profil mis à jour en mémoire\n"
                "   ↓\n"
                "4. 🎭 Système calcule tes genres préférés\n"
                "   ├─ Analyse tous les films aimés\n"
                "   └─ Compte fréquence de chaque genre\n"
                "   ↓\n"
                "5. 💡 Tu vas sur page Recommandations\n"
                "   ├─ get_personalized_recommendations(df, liked, disliked)\n"
                "   ├─ Pour chaque film aimé → KNN trouve voisins\n"
                "   ├─ Score = similarité + bonus genres + bonus note\n"
                "   └─ Exclusion des films dislikés\n"
                "   ↓\n"
                "6. ✨ Affichage top 20 recommandations triées par score\n"
                "```"
            )

            st.info(
                "💡 **Astuce**\n\n"
                "Pour de meilleures recommandations :\n"
                "- ✅ Like au moins **5-10 films** variés\n"
                "- ✅ Dislike les films que tu n'as vraiment **pas aimés**\n"
                "- ✅ Plus tu interagis, plus le système s'améliore !\n\n"
                "Le profil 'Paul' a déjà 30 films pré-remplis pour démonstration."
            )

        
    
    # Vérifier si l'utilisateur est connecté
    if not st.session_state.get('authenticated', False):
        st.warning("⚠️ Vous n'êtes pas connecté")
        st.info("Pour avoir un profil personnalisé sauvegardé, connectez-vous sur **📊 Espace B2B**")
        st.markdown("---")
        st.markdown("**En mode Invité :**")
        st.markdown("- ✅ Vous pouvez utiliser toutes les fonctionnalités")
        st.markdown("- ⚠️ Votre profil sera sauvegardé sous le nom 'invité'")
        st.markdown("- ⚠️ Votre profil sera partagé avec tous les autres visiteurs non connectés")
        st.markdown("---")
        
        # Demander confirmation
        if not st.checkbox("Je comprends et je souhaite continuer en mode Invité"):
            st.stop()
    
    st.markdown("### Gérez vos films vus et améliorez vos recommandations")
    
    # Récupérer l'utilisateur connecté
    current_user = st.session_state.get('authenticated_user', 'invite')
    
    # Afficher l'utilisateur actif
    if current_user != 'invite':
        st.success(f"👤 Profil de **{current_user}**")
    else:
        st.info("👤 Profil **Invité** (partagé)")
    
    st.markdown("---")
    stats = user_manager.get_statistics(current_user)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📚 Films vus", stats['nb_total'])
    
    with col2:
        st.metric("👍 Films aimés", stats['nb_liked'])
    
    with col3:
        st.metric("👎 Films pas aimés", stats['nb_disliked'])
    
    st.markdown("---")
    
    # ==========================================
    # SECTION : AJOUTER UN FILM
    # ==========================================
    
    st.subheader("📝 Ajouter un film vu")
    
    # Barre de recherche améliorée
    col1, col2 = st.columns([4, 1])
    
    with col1:
        search_query = st.text_input(
            "🔍 Cherchez un film que vous avez vu...",
            placeholder="Ex: Les Évadés, Inception, Avatar...",
            key="profile_search",
            help="Cherchez en français ou en anglais !"
        )
    
    with col2:
        prefer_french_profile = st.checkbox("🇫🇷 FR", value=True, key="prefer_french_profile", help="Priorité français")
    
    # Résultats de recherche
    if search_query and len(search_query) >= 2:
        with st.spinner("Recherche en cours..."):
            results, correction, message = find_movies_with_correction(
                search_query, 
                df_movies, 
                max_results=10,
                prefer_french=prefer_french_profile
            )
            
            # Trier les résultats par année décroissante (plus récent d'abord)
            if len(results) > 0 and 'startYear' in results.columns:
                results = results.sort_values('startYear', ascending=False, na_position='last')
            
            if message:
                st.info(message)
            
            if len(results) > 0:
                st.markdown(f"**{len(results)} résultat(s) trouvé(s)**")
                
                for idx, film in results.iterrows():
                    film_id = film.get('tconst')
                    already_rated = user_manager.is_film_already_rated(current_user, film_id)
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        # Utiliser l'affichage optimisé
                        from utils import format_movie_display
                        
                        titre_affichage = format_movie_display(film, show_both_titles=True)
                        note = film.get('note', 0)
                        
                        # Genres
                        if 'genre' in film.index and isinstance(film['genre'], list) and len(film['genre']) > 0:
                            genres_str = " · ".join(film['genre'][:3])
                        else:
                            genres_str = film.get('genres', 'N/A')
                        
                        st.markdown(f"**{titre_affichage}**")
                        st.markdown(f"⭐ {note:.1f}/10 | 🎭 {genres_str}")
                    
                    with col2:
                        # Afficher le statut si déjà noté
                        if already_rated:
                            if already_rated == 'liked':
                                st.success("✅ Déjà aimé")
                            else:
                                st.error("❌ Déjà pas aimé")
                        else:
                            # Boutons pour ajouter
                            col_like, col_dislike = st.columns(2)
                            
                            with col_like:
                                if st.button("👍", key=f"like_{film_id}"):
                                    user_manager.add_film(current_user, film, 'liked')
                                    st.success("Film ajouté aux films aimés !")
                                    st.rerun()
                            
                            with col_dislike:
                                if st.button("👎", key=f"dislike_{film_id}"):
                                    user_manager.add_film(current_user, film, 'disliked')
                                    st.info("Film ajouté aux films pas aimés")
                                    st.rerun()
                    
                    st.markdown("---")
            
            else:
                st.warning("Aucun film trouvé. Essayez une autre recherche en français ou en anglais.")
    
    st.markdown("---")
    
    # ==========================================
    # SECTION : MES FILMS VUS
    # ==========================================
    
    st.subheader("📚 Mes films vus")
    
    # Tabs pour séparer les films aimés et pas aimés
    tab1, tab2 = st.tabs([f"👍 Films aimés ({stats['nb_liked']})", f"👎 Films pas aimés ({stats['nb_disliked']})"])
    
    # Tab Films aimés
    with tab1:
        liked_films = user_manager.get_liked_films(current_user)
        
        if len(liked_films) == 0:
            st.info("Vous n'avez pas encore ajouté de films aimés. Utilisez la barre de recherche ci-dessus pour commencer !")
        else:
            for film_id, film_data in liked_films:
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    titre = film_data.get('titre', 'Titre inconnu')
                    annee = film_data.get('annee', '?')
                    note = film_data.get('note', 0)
                    
                    st.markdown(f"**{titre}** ({annee})")
                    if note:
                        st.markdown(f"⭐ {note:.1f}/10")
                
                with col2:
                    # Boutons de modification
                    col_change, col_delete = st.columns(2)
                    
                    with col_change:
                        if st.button("👎", key=f"change_to_dislike_{film_id}", help="Passer en 'pas aimé'"):
                            user_manager.update_film_rating(current_user, film_id, 'disliked')
                            st.success("Film déplacé vers 'pas aimés'")
                            st.rerun()
                    
                    with col_delete:
                        if st.button("🗑️", key=f"delete_liked_{film_id}", help="Supprimer"):
                            user_manager.remove_film(current_user, film_id)
                            st.success("Film supprimé")
                            st.rerun()
                
                st.markdown("---")
    
    # Tab Films pas aimés
    with tab2:
        disliked_films = user_manager.get_disliked_films(current_user)
        
        if len(disliked_films) == 0:
            st.info("Aucun film dans cette liste.")
        else:
            for film_id, film_data in disliked_films:
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    titre = film_data.get('titre', 'Titre inconnu')
                    annee = film_data.get('annee', '?')
                    note = film_data.get('note', 0)
                    
                    st.markdown(f"**{titre}** ({annee})")
                    if note:
                        st.markdown(f"⭐ {note:.1f}/10")
                
                with col2:
                    # Boutons de modification
                    col_change, col_delete = st.columns(2)
                    
                    with col_change:
                        if st.button("👍", key=f"change_to_like_{film_id}", help="Passer en 'aimé'"):
                            user_manager.update_film_rating(current_user, film_id, 'liked')
                            st.success("Film déplacé vers 'aimés'")
                            st.rerun()
                    
                    with col_delete:
                        if st.button("🗑️", key=f"delete_disliked_{film_id}", help="Supprimer"):
                            user_manager.remove_film(current_user, film_id)
                            st.success("Film supprimé")
                            st.rerun()
                
                st.markdown("---")
    
    # ==========================================
    # SECTION : MES PRÉFÉRENCES
    # ==========================================
    
    if stats['nb_liked'] > 0:
        st.markdown("---")
        st.subheader("🎯 Mes préférences")
        
        genres_preferes = stats['genres_preferes']
        
        if genres_preferes:
            st.markdown("**Genres préférés (basés sur vos films aimés) :**")
            st.caption("*Un film peut appartenir à plusieurs genres*")
            
            # Calculer le total des occurrences de genres
            total_genre_count = sum(count for _, count in genres_preferes)
            
            for genre, count in genres_preferes:
                # Pourcentage sur le TOTAL des genres (pas sur nb_liked)
                pourcentage = (count / total_genre_count) * 100
                # Plafonner à 100% pour st.progress (qui accepte seulement 0-1)
                progress_value = min(1.0, pourcentage / 100)
                st.progress(progress_value, text=f"{genre} ({count} films, {pourcentage:.0f}%)")



# ==========================================
# PAGE : CINÉMAS CREUSE
# ==========================================

elif page == "🗺️ Cinémas Creuse":
    st.title("🗺️ Cinémas de la Creuse")
    st.markdown("### Trouvez le cinéma le plus proche avec les films à l'affiche")
    
    # ==========================================
    # EXPANDER PÉDAGOGIQUE : EXPLICATION CARTOGRAPHIE
    # ==========================================
    with st.expander("Comprendre le système de cartographie interactive", expanded=False, icon="🗺️"):
        col1, col2, col3 = st.columns([1, 8, 1])

        with col2:
            st.caption("🎓 Découvrez comment afficher une carte interactive avec Folium et calculer les distances.")

            st.markdown("**📍 Position → 🗺️ Carte Folium → 📏 Calcul distances → 🎬 Affichage cinémas**")
            st.divider()

            # ÉTAPE 1
            st.subheader("📍 Étape 1 — Géolocalisation de l'utilisateur")
            st.markdown(
                "Pour afficher les cinémas les plus proches, on a besoin de ta position.\n\n"
                "**🎯 Deux méthodes de localisation**\n\n"
                "**Méthode 1 : Sélection ville prédéfinie**\n"
                "```python\n"
                "# Dictionnaire des villes principales de la Creuse\n"
                "VILLES_CREUSE = {\n"
                "    'Guéret': [46.1703, 1.8717],          # [latitude, longitude]\n"
                "    'La Souterraine': [46.2380, 1.4887],\n"
                "    'Aubusson': [45.9564, 2.1688],\n"
                "    'Boussac': [46.3508, 2.2142],\n"
                "    # ... autres villes\n"
                "}\n\n"
                "# Dans Streamlit\n"
                "selected_city = st.selectbox('Votre ville', list(VILLES_CREUSE.keys()))\n"
                "user_lat, user_lon = VILLES_CREUSE[selected_city]  # Récupère coordonnées\n"
                "```\n\n"
                "**Méthode 2 : Saisie manuelle coordonnées**\n"
                "```python\n"
                "# Si l'utilisateur choisit 'Autre ville (saisie manuelle)'\n"
                "if selected_city == 'Autre ville (saisie manuelle)':\n"
                "    user_lat = st.number_input('Latitude', value=46.17, format='%.4f')\n"
                "    user_lon = st.number_input('Longitude', value=1.87, format='%.4f')\n"
                "```\n\n"
                "💡 **Comment trouver ses coordonnées GPS ?**\n"
                "→ Google Maps : clic droit sur un point → coordonnées s'affichent\n"
                "→ Format : Latitude (Nord-Sud), Longitude (Est-Ouest)"
            )

            # ÉTAPE 2
            st.subheader("🗺️ Étape 2 — Créer une carte avec Folium")
            st.markdown(
                "**Folium** est une bibliothèque Python pour créer des cartes interactives (basée sur Leaflet.js).\n\n"
                "**🎨 Création de la carte**\n"
                "```python\n"
                "import folium\n"
                "from streamlit_folium import st_folium\n\n"
                "def create_map(center_lat, center_lon, cinemas, user_location=None):\n"
                "    '''Crée une carte Folium interactive'''\n"
                "    \n"
                "    # 1. Créer la carte centrée sur un point\n"
                "    m = folium.Map(\n"
                "        location=[center_lat, center_lon],  # Centre de la carte\n"
                "        zoom_start=10,                       # Niveau de zoom (1=monde, 18=rue)\n"
                "        tiles='OpenStreetMap'                # Style de carte (OSM gratuit)\n"
                "    )\n"
                "    \n"
                "    # 2. Ajouter marqueur utilisateur (bleu)\n"
                "    if user_location:\n"
                "        folium.Marker(\n"
                "            location=user_location,\n"
                "            popup='Votre position',\n"
                "            icon=folium.Icon(color='blue', icon='user')  # Icône bleue\n"
                "        ).add_to(m)\n"
                "    \n"
                "    # 3. Ajouter marqueurs cinémas (rouge)\n"
                "    for cinema in cinemas:\n"
                "        folium.Marker(\n"
                "            location=[cinema['lat'], cinema['lon']],\n"
                "            popup=f\"{cinema['nom']} - {cinema['ville']}\",\n"
                "            icon=folium.Icon(color='red', icon='film')  # Icône rouge\n"
                "        ).add_to(m)\n"
                "    \n"
                "    return m\n\n"
                "# Affichage dans Streamlit\n"
                "map_obj = create_map(46.17, 1.87, CINEMAS, user_location=[46.17, 1.87])\n"
                "st_folium(map_obj, width=800, height=500)  # Affiche carte interactive\n"
                "```\n\n"
                "💡 **Autres styles de carte disponibles**\n"
                "- `'OpenStreetMap'` : Classique gratuit\n"
                "- `'CartoDB positron'` : Minimaliste clair\n"
                "- `'CartoDB dark_matter'` : Mode sombre"
            )

            # ÉTAPE 3
            st.subheader("📏 Étape 3 — Calcul de distance (formule de Haversine)")
            st.markdown(
                "Pour trier les cinémas du plus proche au plus loin, on calcule la distance **à vol d'oiseau**.\n\n"
                "**🌍 Formule de Haversine**\n"
                "```python\n"
                "import math\n\n"
                "def calculate_cinema_distance(user_lat, user_lon, cinema_lat, cinema_lon):\n"
                "    '''Calcule distance en km entre deux points GPS (formule Haversine)'''\n"
                "    \n"
                "    # Rayon de la Terre en km\n"
                "    R = 6371\n"
                "    \n"
                "    # Conversion degrés → radians\n"
                "    lat1, lon1 = math.radians(user_lat), math.radians(user_lon)\n"
                "    lat2, lon2 = math.radians(cinema_lat), math.radians(cinema_lon)\n"
                "    \n"
                "    # Différences\n"
                "    dlat = lat2 - lat1\n"
                "    dlon = lon2 - lon1\n"
                "    \n"
                "    # Formule de Haversine\n"
                "    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2\n"
                "    c = 2 * math.asin(math.sqrt(a))\n"
                "    \n"
                "    distance_km = R * c\n"
                "    \n"
                "    return round(distance_km, 2)\n"
                "```\n\n"
                "**📊 Exemple de calcul**\n"
                "```python\n"
                "# Position utilisateur : Guéret (46.1703, 1.8717)\n"
                "# Cinéma Sénéchal : Guéret (46.1689, 1.8735)\n\n"
                "distance = calculate_cinema_distance(46.1703, 1.8717, 46.1689, 1.8735)\n"
                "print(f'Distance : {distance} km')  # → Distance : 0.18 km (180 mètres)\n"
                "```\n\n"
                "💡 **Pourquoi Haversine ?**\n"
                "La Terre est ronde, pas plate ! La formule prend en compte la courbure terrestre "
                "pour un calcul précis même sur de longues distances."
            )

            st.success(
                "💡 **Intérêt pour le projet Cinéma Creuse**\n\n"
                "Cette page aide les habitants à :\n"
                "- ✅ Trouver le cinéma **le plus proche** rapidement\n"
                "- ✅ Voir quels films **sont à l'affiche** dans chaque salle\n"
                "- ✅ Planifier leur sortie cinéma en fonction de la **distance** et des **horaires**\n\n"
                "Pour les gérants de cinémas, c'est un outil de **visibilité** qui valorise leur programmation locale !"
            )

        
    
    # ==========================================
    # SECTION 1 : LOCALISATION UTILISATEUR
    # ==========================================
    
    st.subheader("📍 Votre Position")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_city = st.selectbox(
            "Sélectionnez votre ville",
            options=list(VILLES_CREUSE.keys())
        )
    
    with col2:
        show_position = st.checkbox("Afficher sur la carte", value=True)
    
    default_lat, default_lon = VILLES_CREUSE[selected_city]
    
    if selected_city == "Autre ville (saisie manuelle)":
        col_a, col_b = st.columns(2)
        with col_a:
            user_lat = st.number_input("Latitude", value=default_lat, format="%.4f")
        with col_b:
            user_lon = st.number_input("Longitude", value=default_lon, format="%.4f")
    else:
        user_lat, user_lon = default_lat, default_lon
        st.success(f"📍 **{selected_city}** : {user_lat:.4f}, {user_lon:.4f}")
    
    user_location = [user_lat, user_lon] if show_position else None
    
    st.markdown("---")
    
    # ==========================================
    # SECTION 2 : RÉCUPÉRATION FILMS À L'AFFICHE
    # ==========================================
    
    st.subheader("🎬 Films actuellement à l'affiche en France")
    
    with st.spinner("📥 Chargement des films à l'affiche..."):
        
        # Récupérer les films enrichis (avec fallback sur cache si API bloquée)
        films_affiche = get_films_affiche_enrichis()
        
        if len(films_affiche) > 0:
            st.success(f"✅ {len(films_affiche)} films à l'affiche disponibles")
            
            # Assigner aux cinémas (7 films par cinéma)
            cinema_films = assign_films_to_cinemas_enrichis(films_affiche, CINEMAS)
            
        else:
            st.error("❌ Impossible de récupérer les films à l'affiche")
            cinema_films = {}
    
    st.markdown("---")
    
    # ==========================================
    # SECTION 3 : CARTE INTERACTIVE
    # ==========================================
    
    st.subheader("🗺️ Carte Interactive")
    map_obj = create_map(user_location)
    st_folium(map_obj, width=None, height=500)
    
    st.markdown("---")
    
    # ==========================================
    # SECTION 4 : LISTE CINÉMAS TRIÉE PAR DISTANCE
    # ==========================================
    
    st.subheader("🎬 Cinémas les plus proches")
    
    if user_location:
        # Calculer la distance pour chaque cinéma
        cinemas_with_distance = []
        
        for cinema in CINEMAS:
            dist_km = calculate_cinema_distance(cinema, user_location)
            
            cinemas_with_distance.append({
                **cinema,
                'distance_km': dist_km
            })
        
        # TRIER PAR DISTANCE
        cinemas_with_distance.sort(key=lambda x: x['distance_km'])
        
        # AFFICHER
        for idx, cinema in enumerate(cinemas_with_distance, 1):
            
            # Récupérer les films de ce cinéma
            films_cinema = cinema_films.get(cinema['nom'], [])
            nb_films = len(films_cinema)
            
            with st.expander(
                f"#{idx} • 🎬 **{cinema['nom']}** - {cinema['ville']} "
                f"({cinema['distance_km']:.1f} km) • {nb_films} films",
                expanded=(idx == 1)  # Premier cinéma ouvert par défaut
            ):
                # Informations du cinéma
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**📍 Adresse** : {cinema['adresse']}")
                    st.markdown(f"**📞 Téléphone** : {cinema['telephone']}")
                
                with col2:
                    st.metric("Distance", f"{cinema['distance_km']:.1f} km")
                    if cinema['distance_km'] < 10:
                        st.success("🚗 Très proche !")
                    elif cinema['distance_km'] < 30:
                        st.info("🚗 À proximité")
                    else:
                        st.warning("🚗 Un peu éloigné")
                
                st.markdown("---")
                
                # Films à l'affiche pour ce cinéma
                if films_cinema:
                    st.markdown(f"### 🎬 {nb_films} films à l'affiche")
                    
                    # Afficher en grille
                    cols = st.columns(min(4, nb_films))
                    
                    for film_idx, film in enumerate(films_cinema):
                        with cols[film_idx % min(4, nb_films)]:
                            
                            # Les films enrichis ont déjà toutes les infos
                            st.image(film['poster_url'], use_container_width=True)
                            
                            # Titre
                            title = film.get('titre', 'Sans titre')
                            st.markdown(f"**{title[:30]}{'...' if len(title) > 30 else ''}**")
                            
                            # Note + nombre de votes
                            note = film.get('note', 0)
                            st.markdown(f"⭐ {note:.1f}/10")
                            if film.get('nb_votes', 0) > 0:
                                st.caption(f"📊 {film['nb_votes']:,} votes")
                            
                            # Année
                            if film.get('annee'):
                                st.caption(f"📅 {film['annee']}")
                            
                            # Durée
                            if film.get('duree'):
                                st.caption(f"⏱️ {film['duree']} min")
                            
                            # Genres (traduits en français)
                            genres = film.get('genres', [])
                            if genres:
                                genres_traduits = translate_genres(genres[:2])
                                st.caption(f"🎭 {', '.join(genres_traduits)}")
                            
                            # EXPANDER pour les détails complets
                            with st.expander("📄 Plus d'infos"):
                                # Synopsis complet (SANS image)
                                st.markdown("**📝 Synopsis**")
                                st.markdown(film['synopsis'])
                                
                                st.markdown("---")
                                
                                # Réalisateur
                                if film.get('realisateur') and film['realisateur'] != 'Inconnu':
                                    st.markdown(f"**🎬 Réalisateur** : {film['realisateur']}")
                                
                                # Acteurs
                                if film.get('acteurs'):
                                    st.markdown(f"**👥 Acteurs** : {', '.join(film['acteurs'][:5])}")
                                
                                # Durée
                                if film.get('duree'):
                                    st.markdown(f"**⏱️ Durée** : {film['duree']} min")
                                
                                # Genres complets
                                if genres:
                                    st.markdown(f"**🎭 Genres** : {', '.join(genres)}")
                                
                                # Date de sortie
                                if film.get('date_sortie'):
                                    st.markdown(f"**📅 Sortie** : {film['date_sortie']}")
                                
                                # Langue originale
                                if film.get('langue_originale'):
                                    st.markdown(f"**🌍 Langue** : {film['langue_originale'].upper()}")
                                
                                # Titre original si différent
                                if film.get('titre_original') and film['titre_original'] != film['titre']:
                                    st.caption(f"*Titre original : {film['titre_original']}*")
                
                else:
                    st.info("📭 Pas d'informations sur les films à l'affiche pour ce cinéma")
                    st.caption("Les films sont assignés aléatoirement parmi ceux à l'affiche en France")
    
    else:
        # Sans localisation, afficher liste normale (non triée)
        st.info("📍 Sélectionnez votre position pour voir les cinémas triés par distance")
        
        for cinema in CINEMAS:
            films_cinema = cinema_films.get(cinema['nom'], [])
            nb_films = len(films_cinema)
            
            with st.expander(f"🎬 {cinema['nom']} - {cinema['ville']} • {nb_films} films"):
                st.markdown(f"**📍 Adresse** : {cinema['adresse']}")
                st.markdown(f"**📞 Téléphone** : {cinema['telephone']}")
                
                if films_cinema:
                    st.markdown("---")
                    st.markdown(f"### 🎬 {nb_films} films à l'affiche")
                    
                    # Afficher les films avec leurs infos
                    for film in films_cinema:
                        title = film.get('titre', 'Sans titre')
                        note = film.get('note', 0)
                        st.markdown(f"- **{title}** (⭐ {note:.1f}/10)")



# ==========================================
# PAGE : ACTIVITÉS ANNEXES
# ==========================================

elif page == "🎭 Activités Annexes":
    st.title("🎭 Activités Annexes")
    st.markdown("### Événements et animations autour du cinéma")
    
    # ==========================================
    # EXPANDER PÉDAGOGIQUE : EXPLICATION SYSTÈME ÉVÉNEMENTS
    # ==========================================
    with st.expander("Comprendre le système dévénements culturels", expanded=False, icon="🎭"):
        col1, col2, col3 = st.columns([1, 8, 1])

        with col2:
            st.caption("🎓 Découvrez comment gérer et afficher des événements culturels annexes.")

            st.markdown("**📝 Données statiques → 🔍 Filtrage → 📅 Tri → 🎭 Affichage**")
            st.divider()

            # CONCEPT
            st.subheader("🎯 Concept : Valoriser l'expérience cinéma")
            st.markdown(
                "Au-delà du film, les cinémas proposent des **activités complémentaires** :\n"
                "- 🎬 Projections spéciales (avant-premières, ciné-débats)\n"
                "- 🎤 Rencontres avec réalisateurs/acteurs\n"
                "- 🎨 Ateliers créatifs (stop-motion, montage)\n"
                "- 🎶 Ciné-concerts\n"
                "- 🧘 Séances bien-être (ciné-yoga)\n\n"
                "Cette page centralise tous ces événements en un seul endroit."
            )

            # ÉTAPE 1
            st.subheader("📝 Étape 1 — Structure des données")
            st.markdown(
                "Les activités sont stockées dans une **liste de dictionnaires Python**.\n\n"
                "**🗂️ Fichier utils.py**\n"
                "```python\n"
                "ACTIVITES_ANNEXES = [\n"
                "    {\n"
                "        'type': 'Ciné-débat',\n"
                "        'titre': 'Soirée Christopher Nolan',\n"
                "        'description': 'Projection Oppenheimer + débat avec critique cinéma',\n"
                "        'cinema': 'Sénéchal (Guéret)',\n"
                "        'date': '2026-02-15',\n"
                "        'horaire': '20h00',\n"
                "        'tarif': '12€'\n"
                "    },\n"
                "    # ... autres activités\n"
                "]\n"
                "```\n\n"
                "💡 **Pourquoi des dictionnaires ?**\n"
                "Facile à lire, modifier, et parcourir. Pour une vraie app, on utiliserait une base de données."
            )

            # ÉTAPE 2
            st.subheader("🔍 Étape 2 — Système de filtrage")
            st.markdown(
                "Les utilisateurs peuvent filtrer par **type d'activité**.\n\n"
                "**🎚️ Interface Streamlit**\n"
                "```python\n"
                "# 1. Extraire tous les types uniques\n"
                "all_types = list(set([a['type'] for a in ACTIVITES_ANNEXES]))\n\n"
                "# 2. Multiselect pour sélection multiple\n"
                "filter_type = st.multiselect(\n"
                "    'Filtrer par type',\n"
                "    options=all_types,\n"
                "    default=[]  # Rien sélectionné = tout affiché\n"
                ")\n\n"
                "# 3. Filtrer la liste\n"
                "filtered_activities = [\n"
                "    a for a in ACTIVITES_ANNEXES \n"
                "    if a['type'] in filter_type\n"
                "]\n"
                "```"
            )

            # ÉTAPE 3
            st.subheader("📅 Étape 3 — Tri par date")
            st.markdown(
                "Les événements peuvent être triés chronologiquement.\n\n"
                "**🔀 Tri avec sorted()**\n"
                "```python\n"
                "sort_by_date = st.checkbox('Trier par date', value=True)\n\n"
                "if sort_by_date:\n"
                "    filtered_activities = sorted(\n"
                "        filtered_activities,\n"
                "        key=lambda x: x['date']  # Utilise 'date' pour comparer\n"
                "    )\n"
                "```\n\n"
                "💡 **Pourquoi lambda ?**\n"
                "`lambda x: x['date']` dit : 'pour chaque activité x, utilise x['date'] pour le tri'"
            )

            # ÉTAPE 4
            st.subheader("🎭 Étape 4 — Affichage avec expanders")
            st.markdown(
                "Chaque activité s'affiche dans un `st.expander()`.\n\n"
                "**📦 Boucle d'affichage**\n"
                "```python\n"
                "for activity in filtered_activities:\n"
                "    with st.expander(f\"{activity['type']} - {activity['titre']}\"):\n"
                "        col1, col2 = st.columns([2, 1])\n"
                "        \n"
                "        with col1:\n"
                "            st.markdown(f\"**📝 Description** : {activity['description']}\")\n"
                "            st.markdown(f\"**🎬 Cinéma** : {activity['cinema']}\")\n"
                "        \n"
                "        with col2:\n"
                "            st.markdown(f\"### {activity['tarif']}\")\n"
                "            if st.button('Réserver', key=f\"book_{activity['titre']}\"):\n"
                "                st.success('Réservation simulée !')\n"
                "```\n\n"
                "💡 **Importance du key**\n"
                "`key=f'book_{title}'` donne un ID unique à chaque bouton (sinon Streamlit confond)"
            )

            st.info(
                "💡 **Valeur ajoutée pour les cinémas**\n\n"
                "Cette page permet aux cinémas de :\n"
                "- ✅ **Diversifier leurs revenus** (ateliers payants, événements)\n"
                "- ✅ **Fidéliser le public** (créer une communauté)\n"
                "- ✅ **Se différencier** de la concurrence streaming\n"
                "- ✅ **Attirer de nouveaux publics** (enfants, seniors)\n\n"
                "Les activités annexes sont un **levier majeur** pour la survie des cinémas ruraux !"
            )

        
    
    # Filtres
    col1, col2 = st.columns([2, 1])
    
    with col1:
        filter_type = st.multiselect(
            "Filtrer par type",
            options=list(set([a['type'] for a in ACTIVITES_ANNEXES])),
            default=[]
        )
    
    with col2:
        sort_by_date = st.checkbox("Trier par date", value=True)
    
    st.markdown("---")
    
    # Filtrer
    filtered_activities = ACTIVITES_ANNEXES
    if filter_type:
        filtered_activities = [a for a in filtered_activities if a['type'] in filter_type]
    
    # Trier
    if sort_by_date:
        filtered_activities = sorted(filtered_activities, key=lambda x: x['date'])
    
    # Afficher
    if len(filtered_activities) == 0:
        st.info("Aucune activité ne correspond aux critères")
    else:
        for activity in filtered_activities:
            with st.expander(f"{activity['type']} - {activity['titre']}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**📝 Description** : {activity['description']}")
                    st.markdown(f"**🎬 Cinéma** : {activity['cinema']}")
                    st.markdown(f"**📅 Date** : {activity['date']}")
                    st.markdown(f"**🕐 Horaire** : {activity['horaire']}")
                
                with col2:
                    st.markdown(f"### {activity['tarif']}")
                    st.button(f"Réserver", key=f"book_{activity['titre']}")



# ==========================================
# PAGE : ESPACE B2B
# ==========================================

elif page == "📊 Espace B2B":
    if st.button("🚪 Se déconnecter"):
        st.session_state.authenticated = False
        st.rerun()
        
    st.title("Espace B2B - Votre cinéma en Creuse")
    
    if not check_password():
        st.stop()
    
    # ==========================================
    # EXPANDER PÉDAGOGIQUE : EXPLICATION ANALYSE BUSINESS
    # ==========================================
    with st.expander("Comprendre lanalyse de marché B2B", expanded=False, icon="📊"):
        col1, col2, col3 = st.columns([1, 8, 1])

        with col2:
            st.caption("🎓 Découvrez comment analyser le marché du cinéma avec des données réelles et des visualisations.")

            st.markdown("**📊 Données Excel → 🔍 Analyse → 📈 Visualisations → 💡 Insights business**")
            st.divider()

            # CONCEPT
            st.subheader("🎯 Objectif : Aide à la décision pour gérants")
            st.markdown(
                "L'**Espace B2B** (Business to Business) est réservé aux professionnels du cinéma.\n\n"
                "**🎬 Qui utilise cette page ?**\n"
                "- Gérants de cinémas indépendants\n"
                "- Responsables de programmation\n"
                "- Décideurs investissant dans des salles rurales\n\n"
                "**🎯 Objectif**\n"
                "Fournir des **analyses chiffrées** pour prendre de meilleures décisions :\n"
                "- Qui est mon public cible ? (âge, CSP, habitudes)\n"
                "- Quels sont mes concurrents ? (streaming, autres cinémas)\n"
                "- Quelle programmation optimiser ? (genres, durées)\n"
                "- Quelles activités annexes développer ?"
            )

            # ÉTAPE 1
            st.subheader("📊 Étape 1 — Structure des données Excel")
            st.markdown(
                "Toutes les données proviennent d'un **fichier Excel multi-feuilles**.\n\n"
                "**📁 Fichier : `Cinemas_existants_creuse.xlsx`**\n"
                "```python\n"
                "# Chargement avec pandas\n"
                "data = {\n"
                "    'cine_csp_g': pd.read_excel(excel_path, sheet_name='Cine_CSP_Global'),\n"
                "    'pop_c': pd.read_excel(excel_path, sheet_name='Population_creuse'),\n"
                "    'movies_type_g': pd.read_excel(excel_path, sheet_name='movies_type_shares'),\n"
                "    # ... 11 feuilles au total\n"
                "}\n"
                "```\n\n"
                "**📋 Exemples de feuilles**\n"
                "- `Population_creuse` : Pyramide des âges par tranche\n"
                "- `Cine_CSP_Global` : Fréquentation par CSP\n"
                "- `movies_type_shares` : Parts de marché par genre\n"
                "- `prix_mensuel` : Comparaison streaming vs cinéma"
            )

            # ÉTAPE 2
            st.subheader("📈 Étape 2 — Création de graphiques personnalisés")
            st.markdown(
                "Toutes les visualisations utilisent `create_styled_barplot()` depuis utils.py.\n\n"
                "**🎨 Fonction générique**\n"
                "```python\n"
                "def create_styled_barplot(data, x, y, hue=None, title='',\n"
                "                         palette=None, show_values=False):\n"
                "    '''Crée un barplot avec la palette Creuse'''\n"
                "    \n"
                "    fig, ax = plt.subplots(figsize=(10,6))\n"
                "    sns.barplot(data=data, x=x, y=y, hue=hue, palette=palette, ax=ax)\n"
                "    ax.set_title(title, fontsize=14, fontweight='bold')\n"
                "    \n"
                "    if show_values:\n"
                "        for container in ax.containers:\n"
                "            ax.bar_label(container, fmt='%.1f', padding=3)\n"
                "    \n"
                "    return fig, ax\n"
                "```\n\n"
                "**📊 Exemple d'utilisation**\n"
                "```python\n"
                "fig, ax = create_styled_barplot(\n"
                "    data=data['cine_csp_g'],\n"
                "    x='CSP',\n"
                "    y='Part des entrées (%)',\n"
                "    title='Fréquentation par CSP',\n"
                "    palette=PALETTE_CREUSE['gradient'],\n"
                "    show_values=True\n"
                ")\n"
                "st.pyplot(fig)\n"
                "```"
            )

            # ÉTAPE 3
            st.subheader("🔄 Étape 3 — Navigation entre graphiques")
            st.markdown(
                "Pour ne pas surcharger la page, on utilise un **système de carrousel**.\n\n"
                "**🎠 Système de navigation**\n"
                "```python\n"
                "# 1. Définir liste de graphiques\n"
                "graphs = [\n"
                "    {'title': '👥 Structure population', 'key': 'population'},\n"
                "    {'title': '💰 Evolution recettes', 'key': 'revenues'}\n"
                "]\n\n"
                "# 2. Initialiser index dans session_state\n"
                "if 'graph_index_tab1' not in st.session_state:\n"
                "    st.session_state.graph_index_tab1 = 0\n\n"
                "# 3. Boutons Précédent/Suivant\n"
                "if st.button('◀ Précédent'):\n"
                "    st.session_state.graph_index_tab1 = \\\n"
                "        (st.session_state.graph_index_tab1 - 1) % len(graphs)\n"
                "    st.rerun()\n\n"
                "# 4. Afficher graphique actuel\n"
                "current = graphs[st.session_state.graph_index_tab1]\n"
                "```\n\n"
                "💡 **Astuce modulo %**\n"
                "`(index + 1) % len(graphs)` fait boucler : 0→1→2→0→1→..."
            )

            # ÉTAPE 4
            st.subheader("🪖 Étape 4 — Analyse SWOT")
            st.markdown(
                "**SWOT = Strengths, Weaknesses, Opportunities, Threats**\n\n"
                "Matrice stratégique pour évaluer la situation d'une entreprise.\n\n"
                "**📊 Structure dans Streamlit**\n"
                "```python\n"
                "col1, col2 = st.columns(2)\n\n"
                "with col1:\n"
                "    st.markdown('**💪 Forces**')\n"
                "    st.markdown('- Cinémas de proximité')\n"
                "    \n"
                "    st.markdown('**⚠️ Faiblesses**')\n"
                "    st.markdown('- Baisse de fréquentation')\n\n"
                "with col2:\n"
                "    st.markdown('**🚀 Opportunités**')\n"
                "    st.markdown('- Tourisme culturel')\n"
                "    \n"
                "    st.markdown('**⚡ Menaces**')\n"
                "    st.markdown('- Concurrence streaming')\n"
                "```"
            )

            # RÉCAP
            st.markdown("---")
            st.markdown("**📋 Structure complète de l'Espace B2B**")
            st.markdown(
                "**Tab 1 : Analyse de marché** (3 graphiques)\n"
                "- Pyramide des âges locale\n"
                "- Évolution des attentes européennes\n"
                "- Évolution des recettes (confiserie + pub)\n\n"
                "**Tab 2 : Analyse concurrentielle** (2 graphiques)\n"
                "- Prix streaming vs cinéma (mensuel)\n"
                "- Parts de marché par type de film\n\n"
                "**Tab 3 : Analyse interne** (4 graphiques)\n"
                "- Fréquentation par CSP\n"
                "- Fréquentation par tranche d'âge\n"
                "- Types de films projetés\n"
                "- Programmation mensuelle\n\n"
                "**Tab 4 : SWOT**\n"
                "- Matrice Forces/Faiblesses/Opportunités/Menaces\n\n"
                "**Tab 5 : Export**\n"
                "- Téléchargement CSV des films et cinémas"
            )

            st.success(
                "💡 **Impact business**\n\n"
                "Cette analyse permet aux gérants de :\n"
                "- ✅ **Adapter la programmation** au public local (âge, CSP)\n"
                "- ✅ **Se positionner** face à la concurrence streaming\n"
                "- ✅ **Identifier opportunités** de diversification\n"
                "- ✅ **Justifier investissements** auprès de financeurs (CNC, mairie)\n\n"
                "Les données chiffrées sont **essentielles** pour convaincre et décider !"
            )

        
    
    # Métriques clés
    st.subheader("📊 Métriques clés de votre département")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Population Creuse", "115 527 hab", "−0.4% (5 ans)")
    with col2:
        st.metric("Âge médian", "51.2 ans", "+4.3 ans")
    with col3:
        st.metric("Cinémas actifs", len(CINEMAS))
    
    st.caption("*Source : Insee, recensements de la population 2012, 2017 et 2023*")
    st.markdown("---")
    
    # Onglets de l'étude
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Analyse de marché",
        "📈 Analyse concurrentielle",
        "💡 Analyse interne",
        "🪖 SWOT",
        "📄 Export"
    ])
    
    with tab1:
        # Initialiser l'index
        if 'graph_index_tab1' not in st.session_state:
            st.session_state.graph_index_tab1 = 0

        graphs = [
            {"title": "👥 Consommateurs : Structure de la population locale", "key": "population"},
            {"title": "🗺️ Evolution des attentes des consommateurs européens", "key": "trend"},
            {"title": "💰 Evolution des recettes des cinémas français", "key": "revenues"},
        ]

        # Navigation
        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            if st.button("◀ Précédent", key='prev_tab1', use_container_width=True):
                st.session_state.graph_index_tab1 = (st.session_state.graph_index_tab1 - 1) % len(graphs)
                st.rerun()

        with col2:
            st.markdown(
                f"<div style='text-align: center; padding: 8px; font-size: 16px; font-weight: bold;'>"
                f"{st.session_state.graph_index_tab1 + 1} / {len(graphs)}"
                f"</div>",
                unsafe_allow_html=True
            )

        with col3:
            if st.button("Suivant ▶", key="next_tab1", use_container_width=True):
                st.session_state.graph_index_tab1 = (st.session_state.graph_index_tab1 + 1) % len(graphs)
                st.rerun()

        current = graphs[st.session_state.graph_index_tab1]
        st.markdown(f"### {current['title']}")

        graph_placeholder = st.empty()

        with graph_placeholder.container():
            if current['key'] == "population":
                col1, col2 = st.columns(2)
                
                with col1:
                    # Préparer les données
                    df_pop_long = pd.melt(
                        data['pop_c'], 
                        id_vars='Age', 
                        value_vars=['Men', 'Women'], 
                        var_name='Gender', 
                        value_name='Population'
                    )

                    # Calculer les pourcentages
                    total_pop = df_pop_long['Population'].sum()
                    df_pop_long['Percentage'] = (df_pop_long['Population'] / total_pop * 100
                    )
                    
                    # Graphique
                    fig, ax = create_styled_barplot(
                        data=df_pop_long,
                        x='Age',
                        y='Percentage',
                        hue='Gender',
                        title='Répartition par âge',
                        xlabel="Groupe d'âge",
                        ylabel='Pourcentage (%)',
                        rotation=45,
                        figsize=(10, 6),
                        palette=[PALETTE_CREUSE['bleu'], PALETTE_CREUSE['rouge']],
                        show_values=True,
                        value_format='%.1f%%'
                    )
                    
                    st.pyplot(fig)
                    plt.close(fig)
                
                with col2:
                    # Calculer les pourcentages
                    data['kids_c']['Percentage'] = (data['kids_c']['Total'] / data['kids_c']['Total'].sum()) * 100
                    
                    fig, ax = create_styled_barplot(
                        data=data['kids_c'],
                        x='Family_Type',
                        y='Percentage',
                        title='Type de cellule familiale',
                        xlabel='Type',
                        ylabel='Pourcentage (%)',
                        rotation=45,
                        figsize=(10, 6),
                        palette=PALETTE_CREUSE['gradient'],
                        show_values=True,
                        value_format='%.1f%%'
                    )
                    
                    st.pyplot(fig)
                    plt.close(fig)
                
                st.caption("*Source : Insee, étude 2022*")
                
                st.info("""
                📊 **Constat** : 
                - Population vieillissante avec 60% de plus de 45 ans
                - 55% de couples sans enfants, 30% de couples avec enfants et 15% de cellules monoparentales
                """)
                
                st.success("""
                💡 **Recommandations** :
                - Films classiques et patrimoniaux
                - Séances matinales adaptées
                - Dynamiser l'offre pour attirer une plus grande proportion de jeunes
                """)
                
            elif current['key'] == "trend":
                col1, col2 = st.columns(2)
                
                with col1:
                    try:
                        st.image(
                            DATA_DIR / "images" / "recovery_rates_post_covid.png",
                            caption="Retour en salles, période post-covid"
                        )
                    except:
                        st.warning("📊 Image non disponible : recovery_rates_post_covid.png")
                        st.info("L'image devrait montrer les taux de retour en salle post-COVID")
                
                with col2:
                    try:
                        st.image(
                            DATA_DIR / "images" / "origin_of_films.png",
                            caption="Origine des films visionnés en Europe"
                        )
                    except:
                        st.warning("📊 Image non disponible : origin_of_films.png")
                        st.info("L'image devrait montrer l'origine des films visionnés en Europe")
                
                st.info("""
                📊 **Constat** :
                
                **Baisse de fréquentation en salles**  
                Depuis la pandémie, beaucoup moins de spectateurs se rendent dans les salles de cinéma, surtout en zones rurales.
                
                **Difficultés en zone rurale**  
                Les salles rurales peinent à attirer les spectateurs, accentuant la désertification culturelle hors des villes.
                
                **Reprise urbaine progressive**  
                Dans les villes, la fréquentation des cinémas augmente lentement grâce à des événements spéciaux et des sorties nationales.
                
                **Origine des films**  
                Une majorité des films visionnés sur des plateformes de streaming/location/vente est d'origine américaine et marque une préférence du public pour les blockbusters.
                """)
                
                st.success("""
                💡 **Recommandations** :
                
                **Créer de la valeur ajoutée au cinéma**  
                Apporter une réelle différence dans l'expérience de visionnage pour faire revenir la clientèle pré-covid (fauteuils, son, lumières)
                
                **Ajuster l'offre de films**  
                Bien que les utilisateurs web préfèrent les films américains, continuer à proposer une offre diversifiée
                """)
                
            elif current['key'] == "revenues":
                st.markdown("### Analyse des ventes de confiseries")
                
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.set_style("whitegrid")
                
                # Ligne 1 : Part des spectateurs
                color1 = PALETTE_CREUSE['principal']
                ax.plot(
                    data['candies_c']['Année'],
                    data['candies_c']['Part des spectateurs achetant confiseries/boissons (%)'],
                    color=color1,
                    linewidth=2.5,
                    marker='o',
                    label='Part des spectateurs (%)'
                )
                ax.set_xlabel('Année', fontsize=12, fontweight='bold')
                ax.set_ylabel('Part des spectateurs (%)', fontsize=12, fontweight='bold', color=color1)
                ax.tick_params(axis='y', labelcolor=color1)
                
                # Ligne 2 : Indice CA
                ax2 = ax.twinx()
                color2 = PALETTE_CREUSE['accent']
                ax2.plot(
                    data['candies_c']['Année'],
                    data['candies_c']['Indice CA confiseries (base 2019 = 100)'],
                    color=color2,
                    linewidth=2.5,
                    marker='s',
                    label='Indice CA (base 100)'
                )
                ax2.set_ylabel('Indice CA (base 100)', fontsize=12, fontweight='bold', color=color2)
                ax2.tick_params(axis='y', labelcolor=color2)
                
                # Titre et légende
                ax.set_title(
                    'Évolution des ventes de confiseries et boissons',
                    fontsize=16,
                    fontweight='bold',
                    pad=20
                )
                
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10)
                
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                
                st.pyplot(fig)
                plt.close(fig)
                
                st.caption('Sources: Statista, CNC, Boxoffice Pro')
                
                st.markdown("### Analyse des dépenses publicitaires")
                
                try:
                    st.image(
                        DATA_DIR / "images" / "advertising_expenditures.png",
                        caption="Dépenses publicitaires"
                    )
                except:
                    st.warning("📊 Image non disponible : advertising_expenditures.png")
                    st.info("L'image devrait montrer l'évolution des dépenses publicitaires dans le secteur du cinéma")
                
                st.info("""
                📊 **Constat** :
                
                **Consommation sur place constante**  
                Bien que la fréquentation des cinémas ait diminué depuis la sortie du covid, les habitudes de consommation restent inchangées et les revenus annexes sont constants.
                
                **Revenus publicitaires**  
                Les recettes publicitaires (souvent locales) continuent de diminuer au profit d'internet et de la télévision, canaux qui offrent un reach plus élevé.
                """)
                
                st.success("""
                💡 **Recommandations** :
                
                **Augmenter l'offre sur place**  
                Les clients dépensent facilement (1/2) dans des produits autres que la place de cinéma. Au-delà des confiseries, il faut augmenter l'offre de produits complémentaires (façon Disneyland)
                
                **Compenser la perte de revenus publicitaires**  
                Par la location de salles, pour des événements d'entreprise, etc.
                """)
        
    with tab2:
        # Initialiser l'index
        if 'graph_index_tab2' not in st.session_state:
            st.session_state.graph_index_tab2 = 0

        graphs = [
            {"title": "Programmation généralistes Vs. indépendants", "key": "prog"},
            {"title": "💰 Prix des abonnements", "key": "price"},
        ]

        # Navigation
        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            if st.button("◀ Précédent", key="prev_tab2", use_container_width=True):
                st.session_state.graph_index_tab2 = (st.session_state.graph_index_tab2 - 1) % len(graphs)
                st.rerun()

        with col2:
            st.markdown(
                f"<div style='text-align: center; padding: 8px; font-size: 16px; font-weight: bold;'>"
                f"{st.session_state.graph_index_tab2 + 1} / {len(graphs)}"
                f"</div>",
                unsafe_allow_html=True
            )

        with col3:
            if st.button("Suivant ▶", key="next_tab2", use_container_width=True):
                st.session_state.graph_index_tab2 = (st.session_state.graph_index_tab2 + 1) % len(graphs)
                st.rerun()

        current = graphs[st.session_state.graph_index_tab2]
        st.markdown(f"### {current['title']}")

        graph_placeholder = st.empty()

        with graph_placeholder.container():
            if current['key'] == "prog":
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Camembert
                    fig, ax = plt.subplots(figsize=(8, 8))
                    
                    colors = PALETTE_CREUSE['gradient']
                    
                    wedges, texts, autotexts = ax.pie(
                        data['movies_type_g']['Part des entrées nationales'],
                        labels=data['movies_type_g']['Type de films'],
                        autopct='%1.1f%%',
                        colors=colors,
                        startangle=90,
                        textprops={'fontsize': 10, 'fontweight': 'bold'}
                    )
                    
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontsize(11)
                        autotext.set_fontweight('bold')
                    
                    ax.set_title(
                        'Répartition des types de films',
                        fontsize=14,
                        fontweight='bold',
                        pad=20
                    )
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    st.caption('Sources: CNC, la fréquentation des salles de cinéma 2024')
                
                with col2:
                    # Préparer les données
                    df_prog_melt = data['prog_g'].melt(
                        id_vars='Type de films',
                        value_vars=['Grandes chaînes (multiplexes)', 'Cinémas indépendants / Art & Essai'],
                        var_name='Type de cinéma',
                        value_name='Pourcentage'
                    )
                    
                    # Graphique
                    fig, ax = create_styled_barplot(
                        data=df_prog_melt,
                        x='Type de cinéma',
                        y='Pourcentage',
                        hue='Type de films',
                        title='Programmation généralistes Vs. indépendants',
                        xlabel='Type de cinéma',
                        ylabel='Pourcentage (%)',
                        rotation=0,
                        figsize=(12, 6),
                        palette=PALETTE_CREUSE['gradient'],
                        show_values=True,
                        value_format='%.1f%%'
                    )
                    
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    st.caption('Sources: CNC, bilan de la diffusion des films en salle')
                    
            elif current['key'] == "price":
                # Streaming prices
                df_stream_melt = data['streaming_price'].melt(
                    id_vars='Plateforme',
                    value_vars=['Prix mini mensuel', 'Prix maxi mensuel'],
                    var_name="Type d'abonnement",
                    value_name="Prix"
                )
                
                # Nettoyer
                df_stream_melt = df_stream_melt[df_stream_melt['Prix'] != '-   € ']
                df_stream_melt = df_stream_melt.dropna(subset=['Prix'])
                
                # Convertir en numérique
                if df_stream_melt['Prix'].dtype == 'object':
                    df_stream_melt['Prix'] = (
                        df_stream_melt['Prix']
                        .str.replace('€', '', regex=False)
                        .str.replace(',', '.', regex=False)
                        .str.strip()
                        .astype(float)
                    )
                
                # Graphique
                fig, ax = create_styled_barplot(
                    data=df_stream_melt,
                    x='Plateforme',
                    y='Prix',
                    hue="Type d'abonnement",
                    title='Comparaison des abonnements streaming : mini vs maxi',
                    xlabel='Plateforme de streaming',
                    ylabel='Prix mensuel (€)',
                    rotation=45,
                    figsize=(14, 8),
                    palette=[PALETTE_CREUSE['bleu'], PALETTE_CREUSE['rouge']],
                    show_values=True,
                    value_format='%.2f€'
                )
                
                # Ligne de prix moyen
                prix_moyen = df_stream_melt['Prix'].mean()
                ax.axhline(
                    y=prix_moyen,
                    color=PALETTE_CREUSE['accent'],
                    linestyle='--',
                    linewidth=2,
                    label=f'Prix moyen: {prix_moyen:.2f}€'
                )
                ax.legend()
                
                st.pyplot(fig)
                plt.close(fig)
                
                # Métriques
                col1, col2, col3 = st.columns(3)
                
                prix_mini_moy = df_stream_melt[
                    df_stream_melt["Type d'abonnement"] == 'Prix mini mensuel'
                ]['Prix'].mean()
                
                prix_maxi_moy = df_stream_melt[
                    df_stream_melt["Type d'abonnement"] == 'Prix maxi mensuel'
                ]['Prix'].mean()
                
                col1.metric("Prix moyen mini", f"{prix_mini_moy:.2f}€")
                col2.metric("Prix moyen maxi", f"{prix_maxi_moy:.2f}€")
                col3.metric("Écart moyen", f"{prix_maxi_moy - prix_mini_moy:.2f}€")
                
                st.caption('Sources: ariase.com, pathe.com, ugc.com')
                
                st.markdown("---")
                
                # Comparaison streaming vs cinéma
                df_mensp_melt = data['mensual_price'].melt(
                    id_vars='type',
                    value_vars=['Prix mini mensuel moyen', 'Prix maxi mensuel moyen'],
                    var_name='Classe prix',
                    value_name='Prix mensuel moyen'
                )
                
                df_mensp_melt = df_mensp_melt.dropna(subset=['Prix mensuel moyen'])
                
                # Graphique
                fig, ax = create_styled_barplot(
                    data=df_mensp_melt,
                    x='type',
                    y='Prix mensuel moyen',
                    hue='Classe prix',
                    title='Comparaison streaming vs cinéma : prix mensuels moyens',
                    xlabel='Type de service',
                    ylabel='Prix mensuel moyen (€)',
                    rotation=0,
                    figsize=(10, 6),
                    palette=[PALETTE_CREUSE['bleu'], PALETTE_CREUSE['rouge']],
                    show_values=True,
                    value_format='%.2f€'
                )
                
                # Personnaliser les labels X
                ax.set_xticklabels(['Streaming', 'Cinéma'], fontsize=11)
                
                st.pyplot(fig)
                plt.close(fig)
                
                # Métriques
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📺 Streaming")
                    prix_mini_stream = df_mensp_melt[
                        (df_mensp_melt['type'] == 'streaming') &
                        (df_mensp_melt['Classe prix'] == 'Prix mini mensuel moyen')
                    ]['Prix mensuel moyen'].values[0]
                    
                    prix_maxi_stream = df_mensp_melt[
                        (df_mensp_melt['type'] == 'streaming') &
                        (df_mensp_melt['Classe prix'] == 'Prix maxi mensuel moyen')
                    ]['Prix mensuel moyen'].values[0]
                    
                    st.metric("Prix mini moyen", f"{prix_mini_stream:.2f}€")
                    st.metric("Prix maxi moyen", f"{prix_maxi_stream:.2f}€")
                    st.metric("Écart", f"{prix_maxi_stream - prix_mini_stream:.2f}€")
                
                with col2:
                    st.markdown("### 🎬 Cinéma")
                    prix_mini_cinema = df_mensp_melt[
                        (df_mensp_melt['type'] == 'cinema') &
                        (df_mensp_melt['Classe prix'] == 'Prix mini mensuel moyen')
                    ]['Prix mensuel moyen'].values[0]
                    
                    prix_maxi_cinema = df_mensp_melt[
                        (df_mensp_melt['type'] == 'cinema') &
                        (df_mensp_melt['Classe prix'] == 'Prix maxi mensuel moyen')
                    ]['Prix mensuel moyen'].values[0]
                    
                    st.metric("Prix mini moyen", f"{prix_mini_cinema:.2f}€")
                    st.metric("Prix maxi moyen", f"{prix_maxi_cinema:.2f}€")
                    st.metric("Écart", f"{prix_maxi_cinema - prix_mini_cinema:.2f}€")
                
                st.caption('Sources: ariase.com, pathe.com, ugc.com')
    
    with tab3:
        st.header("💡 Recommandations Stratégiques")
        
        st.markdown("""
        ### 🎯 Service de Recommandation Personnalisé
        
        #### Objectifs
        1. **Adapter l'offre** aux préférences locales
        2. **Fidéliser** le public existant
        3. **Attirer** de nouveaux spectateurs
        4. **Valoriser** le patrimoine cinématographique
        
        #### Fonctionnalités Proposées
        - 🤖 **Algorithme de recommandation** basé sur les préférences
        - 📱 **Application mobile** pour réservation
        - 🎁 **Programme de fidélité** multi-cinémas
        - 📧 **Newsletter personnalisée** hebdomadaire
        - 🎬 **Événements thématiques** mensuels
        
        #### Axes de Développement
        
        **1. Diversification de la Programmation**
        - Films classiques et patrimoine
        - Cinéma d'auteur et Art & Essai
        - Documentaires locaux
        - Séances famille
        
        **2. Activités Complémentaires**
        - Ciné-yoga et bien-être
        - Rencontres avec réalisateurs
        - Ateliers pédagogiques
        - Ciné-concerts
        
        **3. Partenariats Locaux**
        - Offices de tourisme
        - Établissements scolaires
        - Associations culturelles
        - Commerces locaux
        
        #### Conditions de Réussite
        
        ✅ **Adhésion des gérants** et équipes  
        ✅ **Communication efficace** (réseaux sociaux, presse locale)  
        ✅ **Formation du personnel** aux outils numériques  
        ✅ **Suivi régulier** des indicateurs (fréquentation, satisfaction)  
        ✅ **Adaptation continue** aux retours usagers
        """)
        
        st.markdown("---")
        
        st.subheader("📊 Budget Prévisionnel")
        
        budget_data = {
            "Poste": [
                "Développement application",
                "Communication & Marketing",
                "Formation personnel",
                "Équipements numériques",
                "Maintenance annuelle"
            ],
            "Montant": [
                "15 000€",
                "8 000€",
                "3 000€",
                "5 000€",
                "4 000€/an"
            ]
        }
        
        st.table(pd.DataFrame(budget_data))
        
        st.markdown("**Total investissement initial** : **31 000€**")
        st.markdown("**Coût annuel de fonctionnement** : **4 000€**")
    
    with tab4:
        st.header("🪖 Analyse SWOT")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **💪 Forces**
            - Cinémas de proximité
            - Programmation Art & Essai
            - Tarifs attractifs
            - Lien social fort
            """)
            
            st.markdown("""
            **⚠️ Faiblesses**
            - Baisse de fréquentation
            - Équipements vieillissants
            - Offre limitée
            - Concurrence streaming
            """)
        
        with col2:
            st.markdown("""
            **🚀 Opportunités**
            - Tourisme culturel
            - Événements spéciaux
            - Partenariats locaux
            - Diversification activités
            """)
            
            st.markdown("""
            **⚡ Menaces**
            - Vieillissement population
            - Exode rural
            - Netflix, Disney+, etc.
            - Concurrence urbaine
            """)
    
    with tab5:
        st.header("📄 Export des Données")
        
        st.markdown("Téléchargez les données pour analyse externe.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_films = df_movies.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Télécharger les films (CSV)",
                data=csv_films,
                file_name="films_creuse_2026.csv",
                mime="text/csv"
            )
        
        with col2:
            csv_cinemas = pd.DataFrame(CINEMAS).to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Télécharger les cinémas (CSV)",
                data=csv_cinemas,
                file_name="cinemas_creuse_2026.csv",
                mime="text/csv"
            )


# ==========================================
# FOOTER
# ==========================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        🎬 Cinéma Creuse | Projet Wild Code School 2026 | Développé par Paul, Hamidou & Lynda
    </div>
    """,
    unsafe_allow_html=True
)

