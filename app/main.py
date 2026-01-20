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
    display_youtube_video, get_trailers_from_films, check_title_columns,
    UserManager, init_paul_profile_if_needed
)

# ==========================================
# CONFIGURATION
# ==========================================

# Initialiser le gestionnaire de profils
user_manager = UserManager()

# Initialiser le profil Paul si vide (cache de 30 films)
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
# CHEMINS ET CHARGEMENT
# ==========================================

PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data" 

@st.cache_data
def load_excel_data():
    """Charge les données Excel"""
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
    """Charge le dataset IMDb avec support des titres français"""
    imdb_path = DATA_DIR / 'PARQUETS' / 'imdb_complet_avec_cast.parquet'  # ← NOUVEAU FICHIER
    
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
    ["🏠 Accueil", "🎬 Films à l'affiche", "❤️ Mes Films Favoris", "💡 Recommandations", "🗺️ Cinémas Creuse", "🎭 Activités Annexes", "📊 Espace B2B"]
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
    from utils import separer_films_par_statut
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
            1. Allez sur la page **❤️ Mes Films Favoris**
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
                    # Afficher les recommandations avec affiches
                    for idx, film in films_filtered.head(nb_to_show).iterrows():
                        
                        # Enrichir le film avec TMDb pour l'affiche
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
                            
                            # Genres
                            genres = film.get('genre', [])
                            if isinstance(genres, list) and len(genres) > 0:
                                genres_str = ', '.join(genres[:3])
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
                                st.write(synopsis)
                                
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
                            
                            # Genres
                            if 'genre' in movie.index and isinstance(movie['genre'], list) and len(movie['genre']) > 0:
                                genres_str = " · ".join(movie['genre'][:3])
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
                                                    st.write(enriched.get('synopsis', 'Synopsis non disponible'))
                                                    
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
                            
                            # Genres
                            if 'genre' in movie.index and isinstance(movie['genre'], list) and len(movie['genre']) > 0:
                                genres_str = " · ".join(movie['genre'][:3])
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
                                                    st.write(enriched.get('synopsis', 'Synopsis non disponible'))
                                                    
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
    if st.button("🚪 Se déconnecter"):
        st.session_state.authenticated = False
        st.rerun()
        
    st.title("Espace B2B - Votre cinéma en Creuse")
    
    if not check_password():
        st.stop()
    
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
                    total_pop = df_pop_long.groupby('Age')['Population'].sum()
                    df_pop_long['Percentage'] = df_pop_long.apply(
                        lambda row: (row['Population'] / total_pop[row['Age']]) * 100, 
                        axis=1
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
                            r"C:/Users/paulc/Documents/PROJET 2/data/images/recovery_rates_post_covid.png",
                            caption="Retour en salles, période post-covid"
                        )
                    except:
                        st.warning("📊 Image non disponible : recovery_rates_post_covid.png")
                        st.info("L'image devrait montrer les taux de retour en salle post-COVID")
                
                with col2:
                    try:
                        st.image(
                            r"C:/Users/paulc/Documents/PROJET 2/data/images/origin_of_films.png",
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
                        r"C:/Users/paulc/Documents/PROJET 2/data/images/advertising_expenditures.png",
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

