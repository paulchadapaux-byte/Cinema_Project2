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
    get_project_root, enrich_movie_with_tmdb, format_genre,
    safe_get, check_password, create_map, create_styled_barplot,
    get_now_playing_france, match_now_playing_with_imdb,
    assign_films_to_cinemas, calculate_cinema_distance,
    get_movie_details_from_tmdb, get_films_affiche_enrichis,
    assign_films_to_cinemas_enrichis, find_movies_with_correction,
    display_youtube_video, get_trailers_from_films, check_title_columns
)

# Import du gestionnaire de profils
from user_manager import UserManager

# ==========================================
# CONFIGURATION
# ==========================================

# Initialiser le gestionnaire de profils
user_manager = UserManager()

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
# CHEMINS ET CHARGEMENT
# ==========================================

PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "processed"

@st.cache_data
def load_excel_data():
    """Charge les données Excel"""
    excel_path = DATA_DIR / 'Cinemas_existants_creuse.xlsx'
    
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
    """Charge le dataset IMDb avec support des titres français"""
    imdb_path = DATA_DIR / 'imdb_complet_avec_tags'
    
    if not imdb_path.exists():
        st.error(f"❌ Fichier non trouvé : {imdb_path}")
        return None
    
    try:
        df = pd.read_parquet(imdb_path)
        
        # ==========================================
        # GESTION DES COLONNES DE TITRES
        # ==========================================
        
        # Renommer colonnes pour compatibilité
        column_mapping = {
            'primaryTitle': 'titre',
            'averageRating': 'note',
            'runtimeMinutes': 'durée',
            'numVotes': 'votes'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]
        
        # Vérifier la présence de la colonne frenchTitle
        has_french = 'frenchTitle' in df.columns
        
        if has_french:
            french_count = df['frenchTitle'].notna().sum()
            st.sidebar.success(f"🇫🇷 {french_count:,} titres français disponibles")
        else:
            st.sidebar.warning("⚠️ Titres français non disponibles")
        
        # ==========================================
        # CONVERSIONS ET NETTOYAGE
        # ==========================================
        
        # Conversions numériques
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
        # FILTRES QUALITÉ
        # ==========================================
        
        df = df[
            (df.get('note', 0) > 0) &
            (df.get('votes', 0) >= 100) &
            (df.get('durée', 0) >= 60)
        ].copy()
        
        # ==========================================
        # COLONNE D'AFFICHAGE OPTIMISÉE
        # ==========================================
        
        # Créer une colonne pour l'affichage rapide
        from utils import get_display_title
        df['display_title'] = df.apply(
            lambda row: get_display_title(row, prefer_french=True, include_year=False),
            axis=1
        )
        
        df = df.reset_index(drop=True)
        
        # Stats de chargement
        st.sidebar.info(f"📊 {len(df):,} films chargés")
        
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
# FONCTIONS DE RECOMMANDATION
# ==========================================

def get_recommendations_knn(df, movie_index, n=10):
    """Recommandations via KNN"""
    if 'recommandations' not in df.columns:
        return None
    
    try:
        movie = df.iloc[movie_index]
        if 'recommandations' in movie and isinstance(movie['recommandations'], list):
            reco_tconsts = movie['recommandations'][:n]
            reco_df = df[df['tconst'].isin(reco_tconsts)].head(n)
            return reco_df
    except:
        pass
    
    return None


def get_recommendations_by_similarity(df, movie_index, n=10):
    """Recommandations par similarité"""
    movie = df.iloc[movie_index]
    
    movie_genres = movie.get('genre', [])
    if not isinstance(movie_genres, list):
        movie_genres = []
    
    similarities = []
    
    for idx, row in df.iterrows():
        if idx == movie_index:
            continue
        
        similarity_score = 0
        
        # Genres (60%)
        row_genres = row.get('genre', [])
        if not isinstance(row_genres, list):
            row_genres = []
        
        if movie_genres and row_genres:
            common = len(set(movie_genres) & set(row_genres))
            similarity_score += (common / max(len(movie_genres), len(row_genres))) * 0.6
        
        # Note (30%)
        if 'note' in movie and 'note' in row:
            rating_diff = abs(movie.get('note', 0) - row.get('note', 0))
            similarity_score += max(0, (1 - rating_diff/10)) * 0.3
        
        # Année (10%)
        if 'startYear' in movie and 'startYear' in row:
            if pd.notna(movie.get('startYear')) and pd.notna(row.get('startYear')):
                year_diff = abs(movie['startYear'] - row['startYear'])
                similarity_score += max(0, (1 - year_diff/50)) * 0.1
        
        similarities.append((idx, similarity_score))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in similarities[:n]]
    
    return df.iloc[top_indices]


def get_recommendations(df, movie_index, n=10):
    """Génère des recommandations"""
    reco = get_recommendations_knn(df, movie_index, n)
    
    if reco is not None and len(reco) > 0:
        return reco, "KNN"
    
    reco = get_recommendations_by_similarity(df, movie_index, n)
    return reco, "Similarité"


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🎬 Navigation")

page = st.sidebar.radio(
    "Choisir une page",
    ["🏠 Accueil", "🎬 Films à l'affiche", "💡 Recommandations", "👤 Mon Profil", "🗺️ Cinémas Creuse", "🎭 Activités Annexes", "📊 Espace B2B"]
)

st.sidebar.markdown("---")

# Filtres pour page Accueil
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
# SYSTÈME DE CONNEXION DANS LE SIDEBAR
# ==========================================

st.sidebar.subheader("🔐 Connexion")

# Vérifier l'état de connexion
if st.session_state.get('authenticated', False):
    # Utilisateur connecté
    username = st.session_state.get('authenticated_user', 'Utilisateur')
    
    st.sidebar.success(f"👤 **{username}**")
    st.sidebar.caption("Profil personnalisé actif")
    
    # Bouton de déconnexion
    if st.sidebar.button("🚪 Se déconnecter", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.authenticated_user = None
        st.success("Déconnexion réussie")
        st.rerun()

else:
    # Mode invité - Formulaire de connexion
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
# PAGE : ACCUEIL
# ==========================================

if page == "🏠 Accueil":
    st.title("🎬 Cinéma Creuse - Documentation Technique")
    st.markdown("### Architecture et méthodologie du projet")
    
    # ==========================================
    # SECTION 1 : PRÉSENTATION
    # ==========================================
    
    st.info("""
    **Bienvenue sur la plateforme Cinéma Creuse !**
    
    Ce projet combine des **données structurelles** historiques (IMDb) avec des **données conjoncturelles** 
    en temps réel (TMDb) pour offrir une expérience de recommandation de films complète et moderne.
    """)
    
    st.markdown("---")
    
    # ==========================================
    # SECTION 2 : ARCHITECTURE DES DONNÉES
    # ==========================================
    
    st.header("📊 Architecture des données")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🗄️ Données structurelles : IMDb")
        st.success("""
        **Base statique historique**
        
        📁 **Source** : IMDb public datasets
        
        📊 **Contenu** :
        - 140,000+ films catalogués
        - Notes, durées, genres
        - Années 1950-2026
        - Identifiants uniques
        
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
        - Affiches officielles HD
        - Synopsis français
        - Casting et équipe
        
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
    # SECTION 3 : WORKFLOW
    # ==========================================
    
    st.markdown("---")
    st.header("🔄 Workflow de traitement")
    
    # Créer un diagramme de flux
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Couleurs
    c_imdb = '#F5C518'
    c_tmdb = '#01D277'
    c_proc = '#5D8A66'
    c_out = '#2F5233'
    
    # Sources
    ax.add_patch(FancyBboxPatch((0.3, 7.5), 1.8, 1.2, boxstyle="round,pad=0.1", 
                                 fc=c_imdb, ec='black', lw=2))
    ax.text(1.2, 8.1, 'IMDb', ha='center', fontsize=14, fontweight='bold')
    ax.text(1.2, 7.8, '140k films', ha='center', fontsize=9)
    
    ax.add_patch(FancyBboxPatch((0.3, 5.8), 1.8, 1.2, boxstyle="round,pad=0.1", 
                                 fc=c_tmdb, ec='black', lw=2))
    ax.text(1.2, 6.4, 'TMDb API', ha='center', fontsize=14, fontweight='bold')
    ax.text(1.2, 6.1, 'Temps réel', ha='center', fontsize=9)
    
    # Traitement
    ax.add_patch(FancyBboxPatch((3, 7), 2, 1.5, boxstyle="round,pad=0.1", 
                                 fc=c_proc, ec='black', lw=2))
    ax.text(4, 8.1, 'Nettoyage', ha='center', fontsize=12, fontweight='bold', color='white')
    ax.text(4, 7.7, '• Doublons', ha='center', fontsize=8, color='white')
    ax.text(4, 7.4, '• Normalisation', ha='center', fontsize=8, color='white')
    ax.text(4, 7.1, '• Validation', ha='center', fontsize=8, color='white')
    
    ax.add_patch(FancyBboxPatch((3, 5.3), 2, 1.5, boxstyle="round,pad=0.1", 
                                 fc=c_proc, ec='black', lw=2))
    ax.text(4, 6.4, 'Enrichissement', ha='center', fontsize=12, fontweight='bold', color='white')
    ax.text(4, 6, '• Affiches', ha='center', fontsize=8, color='white')
    ax.text(4, 5.7, '• Synopsis', ha='center', fontsize=8, color='white')
    ax.text(4, 5.4, '• Casting', ha='center', fontsize=8, color='white')
    
    # Algorithmes
    ax.add_patch(FancyBboxPatch((6.2, 7.2), 1.6, 1, boxstyle="round,pad=0.1", 
                                 fc='#3498DB', ec='black', lw=2))
    ax.text(7, 7.9, 'KNN', ha='center', fontsize=11, fontweight='bold', color='white')
    ax.text(7, 7.5, 'Recommandations', ha='center', fontsize=8, color='white')
    
    ax.add_patch(FancyBboxPatch((6.2, 5.8), 1.6, 1, boxstyle="round,pad=0.1", 
                                 fc='#3498DB', ec='black', lw=2))
    ax.text(7, 6.5, 'Similarité', ha='center', fontsize=11, fontweight='bold', color='white')
    ax.text(7, 6.1, 'Cosinus', ha='center', fontsize=8, color='white')
    
    # Pages finales
    pages = [
        ('Films', 1.5, 2),
        ('Recommand.', 3.3, 2),
        ('Cinémas', 5.1, 2),
        ('B2B', 6.9, 2)
    ]
    
    for nom, x, y in pages:
        ax.add_patch(FancyBboxPatch((x, y), 1.4, 0.8, boxstyle="round,pad=0.05", 
                                     fc=c_out, ec='black', lw=2))
        ax.text(x + 0.7, y + 0.4, nom, ha='center', fontsize=9, fontweight='bold', color='white')
    
    # Flèches
    arrow = dict(arrowstyle='->', lw=2, color='black')
    ax.annotate('', xy=(3, 7.75), xytext=(2.1, 8), arrowprops=arrow)
    ax.annotate('', xy=(3, 6.1), xytext=(2.1, 6.4), arrowprops=arrow)
    ax.annotate('', xy=(6.2, 7.7), xytext=(5, 7.7), arrowprops=arrow)
    ax.annotate('', xy=(6.2, 6.3), xytext=(5, 6.1), arrowprops=arrow)
    
    # Vers pages
    ax.annotate('', xy=(2.2, 2.4), xytext=(4, 5.3), arrowprops=arrow)
    ax.annotate('', xy=(4, 2.4), xytext=(6.8, 5.8), arrowprops=arrow)
    ax.annotate('', xy=(5.8, 2.4), xytext=(7.2, 5.8), arrowprops=arrow)
    ax.annotate('', xy=(7.6, 2.4), xytext=(4, 5.3), arrowprops=arrow)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # ==========================================
    # SECTION 4 : STATISTIQUES
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
    
    # Top genres
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
    # SECTION 5 : STACK TECHNIQUE
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
    
    # Récupérer les films à l'affiche
    with st.spinner("🎬 Récupération des films..."):
        films_affiche = get_films_affiche_enrichis()
    
    if not films_affiche:
        st.warning("⚠️ Impossible de récupérer les films à l'affiche pour le moment.")
        st.stop()
    
    # Récupérer les trailers disponibles pour les films à l'affiche
    with st.spinner("🎥 Recherche des trailers disponibles..."):
        trailers_disponibles = get_trailers_from_films(films_affiche, max_trailers=5)
    
    # Afficher un trailer si disponible
    if trailers_disponibles:
        st.markdown("### 🎥 Bande-annonce du moment")
        
        # Sélectionner un trailer (le premier avec la meilleure popularité)
        # On pourrait aussi faire random.choice(list(trailers_disponibles.values()))
        films_avec_trailers = [
            (key, info) for key, info in trailers_disponibles.items()
        ]
        
        # Trier par popularité du film
        films_avec_trailers.sort(
            key=lambda x: x[1]['film_data'].get('popularite', 0),
            reverse=True
        )
        
        # Prendre le film le plus populaire avec un trailer
        if films_avec_trailers:
            selected_key, trailer_info = films_avec_trailers[0]
            
            display_youtube_video(
                video_id=trailer_info['video_id'],
                title=trailer_info['titre'],
                director=trailer_info['realisateur'],
                max_width=900
            )
            
            # Afficher des infos sur le film
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
    
    # Séparer les films par statut
    from films_cache import separer_films_par_statut
    films_en_salles, films_bientot = separer_films_par_statut(films_affiche)
    
    st.success(f"✅ {len(films_en_salles)} films en salles • 🔜 {len(films_bientot)} films à venir")
    
    # Tabs pour séparer les sections
    tab1, tab2 = st.tabs([
        f"🎬 Déjà en salles ({len(films_en_salles)})",
        f"🔜 Bientôt disponibles ({len(films_bientot)})"
    ])
    
    # ==========================================
    # TAB 1 : FILMS DÉJÀ EN SALLES
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
                            st.caption(f"🎭 {', '.join(genres[:2])}")
                        
                        with st.expander("📄 Voir les détails"):
                            st.markdown("**📝 Synopsis**")
                            st.write(film['synopsis'])
                            
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
                        st.caption(f"🎭 {', '.join(genres[:2])}")
                    
                    with st.expander("📄 Voir les détails"):
                        st.markdown("**📝 Synopsis**")
                        st.write(film['synopsis'])
                        
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

elif page == "💡 Recommandations":
    st.title("🎬 Système de Recommandation de Films")
    st.markdown("### Découvrez des films qui correspondent à vos goûts")
    
    # Récupérer l'utilisateur actuel (connecté ou invité)
    current_user = st.session_state.get('authenticated_user', 'invite')
    
    # Afficher l'utilisateur
    if current_user != 'invite':
        st.info(f"👤 Profil de **{current_user}**")
    else:
        st.info("👤 Mode Invité - Connectez-vous pour sauvegarder votre profil")
    
    st.markdown("---")
    
    # Charger les films aimés/pas aimés de l'utilisateur
    liked_films = user_manager.get_liked_films(current_user)
    disliked_films = user_manager.get_disliked_films(current_user)
    
    # ==========================================
    # TABS : 2 MODES DE RECOMMANDATION
    # ==========================================
    
    tab1, tab2 = st.tabs([
        f"🎯 Recommandations Personnalisées ({len(liked_films)} films aimés)",
        "🔍 Recherche par Titre ou Acteur"
    ])
    
    # ==========================================
    # TAB 1 : RECOMMANDATIONS BASÉES SUR LE PROFIL
    # ==========================================
    
    with tab1:
        st.markdown("### 🎯 Films recommandés pour vous")
        
        if len(liked_films) == 0:
            st.info("💡 **Aucun film aimé dans votre profil**")
            st.markdown("""
            Pour recevoir des recommandations personnalisées :
            1. Allez sur la page **👤 Mon Profil**
            2. Recherchez des films que vous avez aimés
            3. Cliquez sur 👍 pour les ajouter
            4. Revenez ici pour voir vos recommandations !
            """)
        
        else:
            st.markdown(f"*Basées sur vos **{len(liked_films)} films aimés** et vos genres préférés*")
            
            # Importer la fonction de recommandations
            from utils import get_personalized_recommendations
            
            # Générer les recommandations
            with st.spinner("🎬 Génération de vos recommandations personnalisées..."):
                recommended_films = get_personalized_recommendations(
                    df_movies, 
                    liked_films, 
                    disliked_films, 
                    top_n=20
                )
            
            if len(recommended_films) > 0:
                st.success(f"✨ **{len(recommended_films)} films recommandés** pour vous !")
                
                # Options d'affichage
                col_opt1, col_opt2 = st.columns(2)
                with col_opt1:
                    nb_to_show = st.slider("Nombre de films à afficher", 5, 20, 10, step=5, key="slider_nb_films")
                with col_opt2:
                    min_score = st.slider("Score minimum (%)", 0, 100, 50, step=10, key="slider_score")
                
                # Filtrer par score
                films_filtered = recommended_films[
                    recommended_films.get('score_recommandation', 0) >= min_score
                ]
                
                st.markdown("---")
                
                if len(films_filtered) == 0:
                    st.warning(f"Aucun film avec un score >= {min_score}%. Réduisez le score minimum.")
                else:
                    # Afficher les recommandations
                    for idx, film in films_filtered.head(nb_to_show).iterrows():
                        col1, col2 = st.columns([4, 1])
                        
                        with col1:
                            titre = film['titre']
                            annee = film.get('startYear', '?')
                            note = film.get('note', 0)
                            genres = film.get('genres', '')
                            score_reco = film.get('score_recommandation', 0)
                            
                            st.markdown(f"**{titre}** ({annee})")
                            st.markdown(f"⭐ {note:.1f}/10 | {genres}")
                            
                            # Barre de progression du score de recommandation
                            st.progress(score_reco / 100, text=f"Correspondance : {score_reco:.0f}%")
                        
                        with col2:
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
                                col_like, col_dislike = st.columns(2)
                                with col_like:
                                    if st.button("👍", key=f"tab1_reco_like_{film_id}"):
                                        user_manager.add_film(current_user, film, 'liked')
                                        st.success("Ajouté !")
                                        st.rerun()
                                with col_dislike:
                                    if st.button("👎", key=f"tab1_reco_dislike_{film_id}"):
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
        st.markdown("*Cherchez par titre de film et obtenez des recommandations*")
        
        # Barre de recherche
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            search_query = st.text_input(
                "Entrez le nom d'un film que vous aimez",
                placeholder="Ex: Les Évadés, Inception, Le Seigneur des Anneaux...",
                label_visibility="collapsed",
                help="Vous pouvez chercher en français ou en anglais !",
                key="search_tab2"
            )
        
        with col2:
            prefer_french = st.checkbox("🇫🇷 Priorité français", value=True, help="Prioriser les résultats avec titre français", key="prefer_french_tab2")
        
        with col3:
            search_button = st.button("🔍 Rechercher", use_container_width=True, key="search_btn_tab2")
        
        # Résultats de recherche
        if search_query or search_button:
            
            # Utiliser la fonction de correction orthographique optimisée
            matching_movies, correction, correction_message = find_movies_with_correction(
                search_query, 
                df_movies, 
                max_results=10,
                prefer_french=prefer_french
            )
            
            # Afficher le message de correction si présent
            if correction_message:
                st.info(correction_message)
            
            if len(matching_movies) == 0:
                st.warning(f"❌ Aucun film trouvé pour '{search_query}'")
                st.info("💡 Essayez avec un autre titre, en français ou en anglais, ou une partie du titre")
            
            else:
                st.success(f"✅ {len(matching_movies)} film(s) trouvé(s)")
                
                st.markdown("---")
                st.subheader("📋 Résultats de recherche")
                
                for idx, (_, movie) in enumerate(matching_movies.iterrows()):
                    col1, col2 = st.columns([1, 4])
                    
                    with col1:
                        st.markdown(f"**{idx+1}.**")
                    
                    with col2:
                        # Utiliser la fonction d'affichage optimisée
                        from utils import format_movie_display, get_both_titles
                        
                        display_title = format_movie_display(movie, show_both_titles=True)
                        rating = movie.get('note', 0)
                        votes = movie.get('votes', 0)
                        
                        st.markdown(f"**{display_title}** - ⭐ {rating:.1f}/10")
                        
                        if votes > 0:
                            st.caption(f"🗳️ {votes:,} votes")
                        
                        # Afficher les genres si disponibles
                        if 'genre' in movie.index and isinstance(movie['genre'], list) and len(movie['genre']) > 0:
                            genres_str = " · ".join(movie['genre'][:3])
                            st.caption(f"🎭 {genres_str}")
                        
                        if st.button(f"🎬 Voir les recommandations", key=f"tab2_reco_{idx}"):
                            st.session_state.selected_movie_index = movie.name
                            st.session_state.selected_movie_title = display_title
                            st.rerun()
        
        # Affichage des recommandations
        if 'selected_movie_index' in st.session_state:
            
            selected_idx = st.session_state.selected_movie_index
            selected_title = st.session_state.selected_movie_title
            
            st.markdown("---")
            st.subheader(f"💡 Films similaires à : **{selected_title}**")
            
            with st.spinner("🔄 Génération des recommandations..."):
                reco_df, method = get_recommendations(df_movies, selected_idx, n=8)
            
            st.caption(f"Méthode : {method}")
            
            if len(reco_df) == 0:
                st.warning("Aucune recommandation trouvée")
            
            else:
                # Enrichir avec API TMDb
                enriched_movies = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, (_, movie) in enumerate(reco_df.iterrows()):
                    status_text.text(f"Chargement {i+1}/{len(reco_df)} : {movie['titre']}")
                    progress_bar.progress((i+1) / len(reco_df))
                    
                    enriched = enrich_movie_with_tmdb(movie)
                    enriched_movies.append(enriched)
                
                progress_bar.empty()
                status_text.empty()
                
                # Afficher les films enrichis
                cols = st.columns(4)
                
                for i, film in enumerate(enriched_movies):
                    with cols[i % 4]:
                        st.image(film['poster_url'], use_container_width=True)
                        st.markdown(f"**{film['title'][:30]}{'...' if len(film['title']) > 30 else ''}**")
                        
                        if film['rating']:
                            st.markdown(f"⭐ {film['rating']:.1f}/10")
                        
                        if film['year']:
                            st.caption(f"📅 {film['year']}")
                        
                        if film['director'] != 'Inconnu':
                            st.caption(f"🎬 {film['director'][:20]}")
                        
                        if film['genres']:
                            genres_str = ', '.join(film['genres'][:2])
                            st.caption(f"🎭 {genres_str}")
                        
                        if st.button("📄 Détails", key=f"tab2_details_{i}"):
                            st.session_state.show_detail_index = i
                
                # Détails du film sélectionné
                if 'show_detail_index' in st.session_state:
                    detail_idx = st.session_state.show_detail_index
                    film = enriched_movies[detail_idx]
                    
                    st.markdown("---")
                    st.subheader(f"📄 Détails : {film['title']}")
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.image(film['poster_url'], width=300)
                    
                    with col2:
                        st.markdown(f"### {film['title']}")
                        
                        if film['year']:
                            st.markdown(f"**📅 Année** : {film['year']}")
                        
                        if film['rating']:
                            st.markdown(f"**⭐ Note** : {film['rating']:.1f}/10")
                        
                        if film['runtime']:
                            st.markdown(f"**⏱️ Durée** : {film['runtime']} min")
                        
                        if film['director'] != 'Inconnu':
                            st.markdown(f"**🎬 Réalisateur** : {film['director']}")
                        
                        if film['genres']:
                            st.markdown(f"**🎭 Genres** : {', '.join(film['genres'])}")
                        
                        if film['cast']:
                            st.markdown(f"**👥 Acteurs** : {', '.join(film['cast'])}")
                        
                        st.markdown("---")
                        st.markdown(f"**📝 Synopsis** : {film['synopsis']}")
                    
                    if st.button("❌ Fermer", key="tab2_close_detail"):
                        del st.session_state.show_detail_index
                        st.rerun()


elif page == "👤 Mon Profil":
    st.title("👤 Mon Profil")
    
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
                            
                            # Genres
                            genres = film.get('genres', [])
                            if genres:
                                st.caption(f"🎭 {', '.join(genres[:2])}")
                            
                            # EXPANDER pour les détails complets
                            with st.expander("📄 Plus d'infos"):
                                # Synopsis complet (SANS image)
                                st.markdown("**📝 Synopsis**")
                                st.write(film['synopsis'])
                                
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
    
    st.title("📊 Espace B2B - Votre cinéma en Creuse")
    
    # Vérifier si l'utilisateur est connecté
    if not st.session_state.get('authenticated', False):
        st.warning("🔒 Accès réservé aux utilisateurs connectés")
        st.info("👉 Connectez-vous dans le menu de gauche pour accéder à cette page")
        st.markdown("---")
        st.markdown("**Cette page contient :**")
        st.markdown("- 📊 Analyse démographique de la Creuse")
        st.markdown("- 💰 Analyse économique du marché")
        st.markdown("- 🎬 Données du marché cinéma")
        st.markdown("- 📄 Export des données")
        st.stop()
    
    # Utilisateur connecté - afficher le contenu
    username = st.session_state.get('authenticated_user', 'Utilisateur')
    st.success(f"👤 Connecté en tant que **{username}**")
    
    # Métriques
    st.subheader("📊 Métriques clés")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Population Creuse", "115 527 hab", "−0.4% (5 ans)")
    with col2:
        st.metric("Âge médian", "51.2 ans", "+4.3 ans")
    with col3:
        st.metric("Cinémas actifs", len(CINEMAS))
    with col4:
        st.metric("Films catalogue", f"{len(df_movies):,}")
    
    st.caption("*Source : Insee*")
    st.markdown("---")
    
    # Onglets
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Analyse démographique",
        "💰 Analyse économique", 
        "🎬 Marché cinéma",
        "📄 Export",
        "🔧 Diagnostic"
    ])
    
    with tab1:
        st.header("📊 Analyse démographique")
        
        if data and 'pop_c' in data:
            st.subheader("👥 Structure de la population")
            
            df_pop = data['pop_c']
            
            df_pop_long = pd.melt(
                df_pop, 
                id_vars='Age', 
                value_vars=['Men', 'Women'], 
                var_name='Gender', 
                value_name='Population'
            )
            
            total_pop = df_pop_long.groupby('Age')['Population'].sum()
            df_pop_long['Percentage'] = df_pop_long.apply(
                lambda row: (row['Population'] / total_pop[row['Age']]) * 100, 
                axis=1
            )
            
            fig, ax = create_styled_barplot(
                data=df_pop_long,
                x='Age',
                y='Percentage',
                hue='Gender',
                title='Répartition par âge et genre',
                xlabel="Groupe d'âge",
                ylabel='Pourcentage (%)',
                rotation=45,
                figsize=(12, 6),
                palette=[PALETTE_CREUSE['bleu'], PALETTE_CREUSE['rouge']],
                show_values=True,
                value_format='%.1f%%'
            )
            
            st.pyplot(fig)
            plt.close(fig)
            
            st.markdown("---")
            
            st.info("📊 **Constat** : Population vieillissante avec majorité de +45 ans")
            
            # Diplômes
            if 'dip_c' in data:
                st.subheader("🎓 Niveau de diplôme")
                
                df_dip = data['dip_c']
                
                fig, ax = create_styled_barplot(
                    data=df_dip,
                    x='Diplome',
                    y='Percentage',
                    title='Répartition par niveau de diplôme',
                    xlabel='Niveau de diplôme',
                    ylabel='Pourcentage (%)',
                    rotation=45,
                    figsize=(12, 6),
                    show_values=True
                )
                
                st.pyplot(fig)
                plt.close(fig)
        
        else:
            st.warning("Données démographiques non disponibles")
    
    with tab2:
        st.header("💰 Analyse économique")
        
        if data and 'streaming_price' in data and 'mensual_price' in data:
            st.subheader("💵 Comparaison Prix : Streaming vs Cinéma")
            
            df_streaming = data['streaming_price']
            df_mensual = data['mensual_price']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🎬 Prix Cinéma")
                st.dataframe(df_mensual, use_container_width=True)
            
            with col2:
                st.markdown("### 📺 Prix Streaming")
                st.dataframe(df_streaming, use_container_width=True)
            
            st.markdown("---")
            
            # Comparaison
            st.subheader("📊 Analyse comparative")
            
            avg_cinema = df_mensual['Price'].mean()
            avg_streaming = df_streaming['Price'].mean()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Prix moyen cinéma", f"{avg_cinema:.2f}€")
            with col2:
                st.metric("Prix moyen streaming", f"{avg_streaming:.2f}€")
            with col3:
                diff = ((avg_cinema - avg_streaming) / avg_streaming) * 100
                st.metric("Différence", f"{diff:+.1f}%")
            
            st.info(
                """
                💡 **Insight** : Le cinéma reste plus cher que le streaming, 
                mais offre une expérience unique et sociale que le streaming ne peut pas remplacer.
                """
            )
        
        else:
            st.warning("Données économiques non disponibles")
    
    with tab3:
        st.header("🎬 Marché du cinéma")
        
        if data and 'cine_age_g' in data:
            st.subheader("👥 Fréquentation par âge")
            
            df_age = data['cine_age_g']
            
            fig, ax = create_styled_barplot(
                data=df_age,
                x='Age',
                y='Percentage',
                title='Fréquentation cinéma par tranche d\'âge',
                xlabel='Tranche d\'âge',
                ylabel='Pourcentage (%)',
                rotation=45,
                figsize=(12, 6),
                show_values=True
            )
            
            st.pyplot(fig)
            plt.close(fig)
            
            st.markdown("---")
        
        if data and 'movies_type_g' in data:
            st.subheader("🎭 Préférences de genres")
            
            df_types = data['movies_type_g']
            
            fig, ax = create_styled_barplot(
                data=df_types,
                x='Type',
                y='Percentage',
                title='Répartition des préférences par genre',
                xlabel='Genre de film',
                ylabel='Pourcentage (%)',
                rotation=45,
                figsize=(12, 6),
                show_values=True
            )
            
            st.pyplot(fig)
            plt.close(fig)
        
        st.markdown("---")
        
        # Recommandations stratégiques
        st.subheader("💡 Recommandations Stratégiques")
        
        st.markdown("""
        ### 🎯 Axes de développement
        
        **1. Diversification de l'offre**
        - Films patrimoine pour public senior
        - Séances thématiques (soirées d'auteur, ciné-débat)
        - Événements culturels annexes
        
        **2. Expérience client enrichie**
        - Système de recommandation personnalisée (✅ intégré)
        - Application mobile de réservation
        - Programme de fidélité
        
        **3. Partenariats locaux**
        - Collaboration avec offices de tourisme
        - Partenariats écoles/associations
        - Offres groupées hébergement + cinéma
        
        **4. Communication digitale**
        - Présence réseaux sociaux renforcée
        - Newsletter personnalisée
        - Campagnes ciblées par tranche d'âge
        """)
    
    with tab4:
        st.header("📄 Export des Données")
        
        st.markdown("### Télécharger les données pour analyse externe")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_films = df_movies.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Catalogue Films (CSV)",
                data=csv_films,
                file_name=f"films_creuse_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            if data and 'pop_c' in data:
                csv_pop = data['pop_c'].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Données Population (CSV)",
                    data=csv_pop,
                    file_name=f"population_creuse_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
    
    with tab5:
        st.header("🔧 Diagnostic Technique")
        
        st.markdown("### Vérification des colonnes de titres")
        st.info("Cette section permet de vérifier quelles colonnes de titres sont disponibles dans la base de données et si des titres français sont présents.")
        
        if st.button("🔍 Lancer le diagnostic", use_container_width=True):
            with st.spinner("Analyse en cours..."):
                # Lancer le diagnostic
                results = check_title_columns(df_movies)
                
                # Afficher les résultats
                st.success("✅ Diagnostic terminé")
                st.markdown("---")
                
                # Colonnes de titres disponibles
                st.subheader("📋 Colonnes de titres disponibles")
                if results['title_columns']:
                    for col in results['title_columns']:
                        st.markdown(f"- `{col}`")
                else:
                    st.warning("Aucune colonne de titre trouvée")
                
                st.markdown("---")
                
                # Résultats des tests français
                st.subheader("🇫🇷 Test de recherche de films français")
                
                for query, cols_results in results['french_test_results'].items():
                    st.markdown(f"**Recherche : '{query}'**")
                    
                    found = False
                    for col, result in cols_results.items():
                        if result['count'] > 0:
                            st.success(f"✅ Trouvé dans `{col}` : {result['example']}")
                            found = True
                            break
                    
                    if not found:
                        st.error(f"❌ '{query}' non trouvé dans aucune colonne")
                
                st.markdown("---")
                
                # Échantillons
                st.subheader("📊 Échantillons de titres")
                
                for col, samples in results['samples'].items():
                    with st.expander(f"Colonne : {col}"):
                        for i, sample in enumerate(samples, 1):
                            st.markdown(f"{i}. {sample}")
                
                st.markdown("---")
                
                # Recommandations
                st.subheader("💡 Recommandations")
                
                for rec in results['recommendations']:
                    if rec['type'] == 'success':
                        st.success(rec['message'])
                    elif rec['type'] == 'warning':
                        st.warning(rec['message'])
                    elif rec['type'] == 'error':
                        st.error(rec['message'])
                    else:
                        st.info(rec['message'])
                
                # Guide pour ajouter des titres français
                if not results['has_french_titles']:
                    st.markdown("---")
                    st.markdown("### 📚 Comment ajouter des titres français")
                    
                    st.markdown("""
                    **Option 1 : Table IMDb akas (recommandé)**
                    1. Télécharger : https://datasets.imdbws.com/title.akas.tsv.gz
                    2. Filtrer les titres avec `region = 'FR'`
                    3. Fusionner avec la base principale
                    
                    **Option 2 : Dictionnaire manuel**
                    ```python
                    french_titles = {
                        'tt1411238': 'Bienvenue chez les Ch\\'tis',
                        'tt1675434': 'Intouchables',
                        # ...
                    }
                    df['titre_francais'] = df['tconst'].map(french_titles)
                    ```
                    """)
 

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        🎬 Cinéma Creuse | Projet Wild Code School 2026 | Paul, Hamidou & Lynda
    </div>
    """,
    unsafe_allow_html=True
)

