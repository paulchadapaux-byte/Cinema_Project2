"""
Application Streamlit - Cinéma Creuse
Version complète avec toutes les fonctionnalités + Recommandations
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from streamlit_folium import st_folium

# Imports depuis utils.py
from utils import (
    PALETTE_CREUSE, CINEMAS, VILLES_CREUSE, ACTIVITES_ANNEXES,
    get_project_root, enrich_movie_with_tmdb, format_genre,
    safe_get, check_password, create_map, create_styled_barplot
)

# ==========================================
# CONFIGURATION
# ==========================================

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
    """Charge le dataset IMDb"""
    imdb_path = DATA_DIR / 'imdb_complet_avec_tags'
    
    if not imdb_path.exists():
        st.error(f"❌ Fichier non trouvé : {imdb_path}")
        return None
    
    try:
        df = pd.read_parquet(imdb_path)
        
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
        
        # Conversions
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
        
        # Filtrer qualité
        df = df[
            (df.get('note', 0) > 0) &
            (df.get('votes', 0) >= 100) &
            (df.get('durée', 0) >= 60)
        ].copy()
        
        df = df.reset_index(drop=True)
        return df
        
    except Exception as e:
        st.error(f"Erreur IMDb : {e}")
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
    ["🏠 Accueil", "🎥 Films", "💡 Recommandations", "🗺️ Cinémas Creuse", "🎭 Activités Annexes", "📊 Espace B2B"]
)

st.sidebar.markdown("---")

# Filtres pour pages Accueil et Films
if page in ["🏠 Accueil", "🎥 Films"]:
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
st.sidebar.markdown(f"**📊 {len(df_movies):,} films**")
st.sidebar.markdown("**📅 Année : 2026**")
st.sidebar.markdown("**🎓 Wild Code School**")



# ==========================================
# PAGE : ACCUEIL
# ==========================================

if page == "🏠 Accueil":
    st.title("🎬 Cinéma Creuse - Plateforme de Recommandation")
    st.markdown("### Découvrez les meilleurs films dans les cinémas de la Creuse")
    
    # Métriques
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Films disponibles", f"{len(df_filtered):,}")
    with col2:
        current_year = datetime.now().year
        films_recents = len(df_filtered[df_filtered['startYear'] >= (current_year - 2)])
        st.metric("Films récents", films_recents)
    with col3:
        st.metric("Cinémas", len(CINEMAS))
    with col4:
        avg_rating = df_filtered['note'].mean()
        st.metric("Note moyenne", f"{avg_rating:.1f}/10")
    
    st.markdown("---")
    
    # Films récents
    st.subheader("🎬 Films Récents (2024-2026)")
    
    current_year = datetime.now().year
    films_recents_df = df_filtered[df_filtered['startYear'] >= (current_year - 2)].nlargest(12, 'note')
    
    if len(films_recents_df) > 0:
        cols = st.columns(4)
        
        for idx, (_, film) in enumerate(films_recents_df.iterrows()):
            with cols[idx % 4]:
                st.image(
                    "https://via.placeholder.com/300x450/2F5233/FFFFFF?text=Film",
                    use_container_width=True
                )
                
                title = film.get('titre', 'Sans titre')
                st.markdown(f"**{title[:30]}{'...' if len(title) > 30 else ''}**")
                st.markdown(f"⭐ {film['note']:.1f}/10")
                
                if pd.notna(film.get('startYear')):
                    st.markdown(f"📅 {int(film['startYear'])}")
                
                st.markdown(f"⏱️ {int(film['durée'])} min")
                
                genres = film.get('genre', [])
                if isinstance(genres, list) and genres:
                    st.caption(', '.join(genres[:2]))
    else:
        st.info("Aucun film récent ne correspond aux critères")
    
    st.markdown("---")
    
    # Top 5 films
    st.subheader("🏆 Top 5 Films par Note")
    
    top5 = df_filtered.nlargest(5, 'note')
    
    for idx, (_, film) in enumerate(top5.iterrows(), 1):
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.image(
                f"https://via.placeholder.com/150x225/2F5233/FFFFFF?text=Film",
                width=150
            )
        
        with col2:
            title = film.get('titre', 'Sans titre')
            st.markdown(f"### {idx}. {title}")
            st.markdown(f"**⭐ {film['note']:.1f}/10** • {format_genre(film.get('genre', []))}")
            
            if pd.notna(film.get('startYear')):
                st.markdown(f"📅 Sortie : {int(film['startYear'])}")
        
        if idx < len(top5):
            st.markdown("---")


# ==========================================
# PAGE : FILMS
# ==========================================

elif page == "🎥 Films":
    st.title("🎥 Catalogue de Films")
    st.markdown(f"### {len(df_filtered):,} films disponibles")
    
    # Options tri
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"**{len(df_filtered):,} films** correspondant aux critères")
    
    with col2:
        sort_by = st.selectbox(
            "Trier par",
            ["Note (desc)", "Note (asc)", "Titre (A-Z)", "Titre (Z-A)", "Année (récent)", "Année (ancien)"]
        )
    
    with col3:
        per_page = st.selectbox("Films par page", [12, 24, 48], index=0)
    
    # Tri
    if sort_by == "Note (desc)":
        df_display = df_filtered.sort_values('note', ascending=False)
    elif sort_by == "Note (asc)":
        df_display = df_filtered.sort_values('note', ascending=True)
    elif sort_by == "Titre (A-Z)":
        df_display = df_filtered.sort_values('titre')
    elif sort_by == "Titre (Z-A)":
        df_display = df_filtered.sort_values('titre', ascending=False)
    elif sort_by == "Année (récent)":
        df_display = df_filtered.sort_values('startYear', ascending=False)
    else:
        df_display = df_filtered.sort_values('startYear', ascending=True)
    
    # Pagination
    total_pages = (len(df_display) - 1) // per_page + 1
    
    if 'page_num' not in st.session_state:
        st.session_state.page_num = 1
    
    col_prev, col_page, col_next = st.columns([1, 2, 1])
    
    with col_prev:
        if st.button("⬅️ Précédent") and st.session_state.page_num > 1:
            st.session_state.page_num -= 1
            st.rerun()
    
    with col_page:
        st.markdown(f"**Page {st.session_state.page_num} / {total_pages}**")
    
    with col_next:
        if st.button("Suivant ➡️") and st.session_state.page_num < total_pages:
            st.session_state.page_num += 1
            st.rerun()
    
    st.markdown("---")
    
    # Affichage
    start_idx = (st.session_state.page_num - 1) * per_page
    end_idx = start_idx + per_page
    page_films = df_display.iloc[start_idx:end_idx]
    
    cols = st.columns(4)
    
    for idx, (_, film) in enumerate(page_films.iterrows()):
        with cols[idx % 4]:
            # Image placeholder simple
            st.image(
                "https://via.placeholder.com/300x450/2F5233/FFFFFF?text=Film",
                use_container_width=True
            )
            
            title = film.get('titre', 'Sans titre')
            st.markdown(f"**{title[:30]}{'...' if len(title) > 30 else ''}**")
            st.markdown(f"⭐ {film['note']:.1f}/10")
            
            if pd.notna(film.get('startYear')):
                st.markdown(f"📅 {int(film['startYear'])}")
            
            st.markdown(f"⏱️ {int(film['durée'])} min")
            
            genres = film.get('genre', [])
            if isinstance(genres, list) and genres:
                st.caption(', '.join(genres[:2]))



# ==========================================
# PAGE : RECOMMANDATIONS
# ==========================================

elif page == "💡 Recommandations":
    st.title("🎬 Système de Recommandation de Films")
    st.markdown("### Découvrez des films similaires à vos goûts")
    
    # Barre de recherche
    st.subheader("🔍 Rechercher un film")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Entrez le nom d'un film que vous aimez",
            placeholder="Ex: The Dark Knight, Inception, Avatar...",
            label_visibility="collapsed"
        )
    
    with col2:
        search_button = st.button("🔍 Rechercher", use_container_width=True)
    
    # Résultats de recherche
    if search_query or search_button:
        
        matching_movies = df_movies[
            df_movies['titre'].str.contains(search_query, case=False, na=False)
        ].head(10)
        
        if len(matching_movies) == 0:
            st.warning(f"❌ Aucun film trouvé pour '{search_query}'")
            st.info("💡 Essayez avec un autre titre ou une partie du titre")
        
        else:
            st.success(f"✅ {len(matching_movies)} film(s) trouvé(s)")
            
            st.markdown("---")
            st.subheader("📋 Résultats de recherche")
            
            for idx, (_, movie) in enumerate(matching_movies.iterrows()):
                col1, col2 = st.columns([1, 4])
                
                with col1:
                    st.markdown(f"**{idx+1}.**")
                
                with col2:
                    title = movie['titre']
                    year = int(movie['startYear']) if pd.notna(movie.get('startYear')) else '?'
                    rating = movie.get('note', 0)
                    
                    st.markdown(f"**{title}** ({year}) - ⭐ {rating:.1f}/10")
                    
                    if st.button(f"🎬 Voir les recommandations", key=f"reco_{idx}"):
                        st.session_state.selected_movie_index = movie.name
                        st.session_state.selected_movie_title = title
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
                    
                    if st.button("📄 Détails", key=f"details_{i}"):
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
                
                if st.button("❌ Fermer"):
                    del st.session_state.show_detail_index
                    st.rerun()



# ==========================================
# PAGE : CINÉMAS CREUSE
# ==========================================

elif page == "🗺️ Cinémas Creuse":
    st.title("🗺️ Cinémas de la Creuse")
    st.markdown("### Trouvez le cinéma le plus proche")
    
    # Localisation
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
    
    # Carte
    st.subheader("🗺️ Carte Interactive")
    map_obj = create_map(user_location)
    st_folium(map_obj, width=None, height=500)
    
    st.markdown("---")
    
    # Liste des cinémas
    st.subheader("🎬 Liste des Cinémas")
    
    for cinema in CINEMAS:
        with st.expander(f"🎬 {cinema['nom']} - {cinema['ville']}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**📍 Adresse** : {cinema['adresse']}")
                st.markdown(f"**📞 Téléphone** : {cinema['telephone']}")
            
            with col2:
                if user_location:
                    dist = ((cinema['lat'] - user_location[0])**2 + (cinema['lon'] - user_location[1])**2)**0.5
                    dist_km = dist * 111
                    st.metric("Distance", f"{dist_km:.1f} km")


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
    
    if st.session_state.get('authenticated', False):
        if st.button("🚪 Se déconnecter"):
            st.session_state.authenticated = False
            st.rerun()
    
    st.title("📊 Espace B2B - Votre cinéma en Creuse")
    
    if not check_password():
        st.stop()
    
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
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Analyse démographique",
        "💰 Analyse économique", 
        "🎬 Marché cinéma",
        "📄 Export"
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

