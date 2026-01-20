"""
Fonctions utilitaires pour l'application Streamlit Cinéma Creuse
Inclut les appels API TMDb pour enrichissement des films
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import requests
import streamlit as st
from pathlib import Path
import numpy as np
from difflib import SequenceMatcher
import unicodedata

# ==========================================
# GESTION DES TITRES BILINGUES
# ==========================================

def get_display_title(row, prefer_french=True, include_year=True, fallback_col='primaryTitle'):
    """
    Retourne le meilleur titre à afficher selon la disponibilité
    Priorité : frenchTitle > titre_francais > titre > primaryTitle > originalTitle
    
    Args:
        row: Ligne du DataFrame (pd.Series)
        prefer_french: Si True, privilégie le français quand disponible
        include_year: Si True, ajoute l'année entre parenthèses
        fallback_col: Colonne de fallback si aucun titre trouvé
    
    Returns:
        str: Titre formaté pour l'affichage
    """
    if prefer_french:
        # Priorité aux titres français
        title = None
        if 'frenchTitle' in row.index and pd.notna(row.get('frenchTitle')):
            title = row['frenchTitle']
        elif 'titre_francais' in row.index and pd.notna(row.get('titre_francais')):
            title = row['titre_francais']
        elif 'titre' in row.index and pd.notna(row.get('titre')):
            title = row['titre']
        elif 'primaryTitle' in row.index and pd.notna(row.get('primaryTitle')):
            title = row['primaryTitle']
        elif 'originalTitle' in row.index and pd.notna(row.get('originalTitle')):
            title = row['originalTitle']
        else:
            title = row.get(fallback_col, "Titre inconnu")
    else:
        # Priorité au titre original
        title = None
        if 'originalTitle' in row.index and pd.notna(row.get('originalTitle')):
            title = row['originalTitle']
        elif 'primaryTitle' in row.index and pd.notna(row.get('primaryTitle')):
            title = row['primaryTitle']
        elif 'titre' in row.index and pd.notna(row.get('titre')):
            title = row['titre']
        else:
            title = row.get(fallback_col, "Titre inconnu")
    
    # Ajouter l'année si demandé
    if include_year and 'startYear' in row.index and pd.notna(row.get('startYear')):
        try:
            year = int(row['startYear'])
            title = f"{title} ({year})"
        except:
            pass
    
    return str(title)


def get_both_titles(row):
    """
    Retourne un tuple (titre_francais, titre_original) pour affichage complet
    
    Args:
        row: Ligne du DataFrame (pd.Series)
    
    Returns:
        tuple: (titre_francais, titre_original)
    """
    french = None
    if 'frenchTitle' in row.index and pd.notna(row.get('frenchTitle')):
        french = row['frenchTitle']
    elif 'titre_francais' in row.index and pd.notna(row.get('titre_francais')):
        french = row['titre_francais']
    elif 'titre' in row.index and pd.notna(row.get('titre')):
        french = row['titre']
    
    original = None
    if 'originalTitle' in row.index and pd.notna(row.get('originalTitle')):
        original = row['originalTitle']
    elif 'primaryTitle' in row.index and pd.notna(row.get('primaryTitle')):
        original = row['primaryTitle']
    
    return french, original


def format_movie_display(row, show_both_titles=True):
    """
    Formatte l'affichage complet d'un film avec titre FR + original + année
    
    Args:
        row: Ligne du DataFrame
        show_both_titles: Si True, affiche titre FR (titre original, année)
    
    Returns:
        str: Titre formatté
    
    Exemple:
        "Les Évadés (The Shawshank Redemption, 1994)"
    """
    french, original = get_both_titles(row)
    year = ""
    
    if 'startYear' in row.index and pd.notna(row.get('startYear')):
        try:
            year = int(row['startYear'])
        except:
            pass
    
    if show_both_titles and french and original and french != original:
        # Les deux titres sont différents, afficher les deux
        if year:
            return f"{french} ({original}, {year})"
        else:
            return f"{french} ({original})"
    elif french:
        # Seulement le français
        if year:
            return f"{french} ({year})"
        else:
            return french
    elif original:
        # Seulement l'original
        if year:
            return f"{original} ({year})"
        else:
            return original
    else:
        return "Titre inconnu"


# ==========================================
# CONSTANTES GLOBALES
# ==========================================

# Palette de couleurs
PALETTE_CREUSE = {
    'principal': '#2F5233',
    'secondaire': '#5D8A66', 
    'accent': '#D4AF37',
    'neutre': '#34495E',
    'bleu': '#3498DB',
    'rouge': '#E74C3C',
    'gradient': ['#2F5233', '#5D8A66', '#8CB369', '#D4AF37']
}

# Configuration TMDb
TMDB_API_KEY = "a8617cdd3b93f8a353f24a1843ccaafb"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

# Identifiants admin
ADMIN_CREDENTIALS = {
    "paul": "WCS26",
    "hamidou": "WCS26",
    "lynda": "WCS26"
}

# Cinémas de la Creuse
CINEMAS = [
    {
        "nom": "Cinéma Le Sénéchal",
        "ville": "Guéret",
        "adresse": "1 Rue du Sénéchal, 23000 Guéret",
        "lat": 46.1710,
        "lon": 1.8716,
        "telephone": "05 55 52 12 50"
    },
    {
        "nom": "Cinéma Eden",
        "ville": "La Souterraine",
        "adresse": "Place Saint-Jacques, 23300 La Souterraine",
        "lat": 46.2376,
        "lon": 1.4879,
        "telephone": "05 55 63 01 77"
    },
    {
        "nom": "Cinéma Colbert",
        "ville": "Aubusson",
        "adresse": "Grande Rue, 23200 Aubusson",
        "lat": 45.9569,
        "lon": 2.1684,
        "telephone": "05 55 66 13 88"
    },
    {
        "nom": "Cinéma Claude Miller",
        "ville": "Bourganeuf",
        "adresse": "Place de l'Hôtel de Ville, 23400 Bourganeuf",
        "lat": 45.9514,
        "lon": 1.7569,
        "telephone": "05 55 64 08 27"
    },
    {
        "nom": "Cinéma Alpha",
        "ville": "Évaux-les-Bains",
        "adresse": "Rue de Rentière, 23110 Évaux-les-Bains",
        "lat": 46.1729,
        "lon": 2.4886,
        "telephone": "05 55 65 50 02"
    },
    {
        "nom": "Cinéma Le Marchois",
        "ville": "La Courtine",
        "adresse": "Rue des Deux Frères, 23100 La Courtine",
        "lat": 45.7046,
        "lon": 2.2679,
        "telephone": "05 55 67 21 15"
    },
    {
        "nom": "Salle des Fêtes (Cinéma)",
        "ville": "Dun-le-Palestel",
        "adresse": "Salle des Fêtes, 23800 Dun-le-Palestel",
        "lat": 46.3053,
        "lon": 1.6665,
        "telephone": "05 55 89 01 23"
    }
]

# Villes de la Creuse
VILLES_CREUSE = {
    "Guéret": (46.1703, 1.8717),
    "La Souterraine": (46.2392, 1.5111),
    "Aubusson": (45.9567, 2.1681),
    "Bourganeuf": (45.9545, 1.7547),
    "Évaux-les-Bains": (46.1729, 2.4886),
    "Boussac": (46.3494, 2.2136),
    "Dun-le-Palestel": (46.3053, 1.6665),
    "La Courtine": (45.7046, 2.2679),
    "Felletin": (45.8828, 2.1742),
    "Ahun": (46.0833, 2.0500),
    "Autre ville (saisie manuelle)": (46.17, 1.87)
}

# Activités annexes
ACTIVITES_ANNEXES = [
    {
        "type": "Yoga",
        "titre": "Ciné-Yoga : Détente avant séance",
        "description": "Séance de yoga doux avant la projection du soir",
        "cinema": "Le Sénéchal",
        "date": "2026-02-15",
        "horaire": "18h30 - 19h30",
        "tarif": "12€ (séance + film)"
    },
    {
        "type": "Conférence",
        "titre": "Rencontre avec le réalisateur",
        "description": "Échange avec Denis Villeneuve autour de 'Beyond the Stars'",
        "cinema": "Ciné Bourse",
        "date": "2026-02-20",
        "horaire": "20h00",
        "tarif": "Gratuit avec billet du film"
    },
    {
        "type": "Avant-première",
        "titre": "Avant-première : Echoes of Tomorrow",
        "description": "Découvrez le film en exclusivité avec cocktail d'accueil",
        "cinema": "Le Turenne",
        "date": "2026-02-25",
        "horaire": "19h00",
        "tarif": "15€"
    },
    {
        "type": "Atelier",
        "titre": "Atelier cinéma pour enfants",
        "description": "Initiation au cinéma d'animation (8-12 ans)",
        "cinema": "Le Sénéchal",
        "date": "2026-03-01",
        "horaire": "14h00 - 17h00",
        "tarif": "10€"
    },
    {
        "type": "Concert",
        "titre": "Ciné-Concert : Musique de film",
        "description": "Concert live accompagnant la projection",
        "cinema": "Ciné Bourse",
        "date": "2026-03-10",
        "horaire": "20h30",
        "tarif": "18€"
    },
    {
        "type": "Débat",
        "titre": "Ciné-Débat : L'écologie au cinéma",
        "description": "Discussion autour des enjeux environnementaux au cinéma",
        "cinema": "Cinéma Fressignes",
        "date": "2026-03-15",
        "horaire": "18h00",
        "tarif": "Gratuit"
    }
]


# ==========================================
# FONCTIONS API TMDb
# ==========================================

def search_tmdb_by_title(title, year=None):
    """
    Recherche un film sur TMDb par titre
    
    Args:
        title: Titre du film
        year: Année de sortie (optionnel)
    
    Returns:
        dict avec résultat de recherche, ou None
    """
    try:
        url = f"{TMDB_BASE_URL}/search/movie"
        params = {
            "api_key": TMDB_API_KEY,
            "language": "fr-FR",
            "query": title,
            "page": 1
        }
        
        if year:
            params["year"] = year
        
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('results') and len(data['results']) > 0:
                return data['results'][0]
        
        return None
        
    except Exception as e:
        print(f"Erreur recherche TMDb : {e}")
        return None


@st.cache_data(ttl=86400)
def get_movie_details_from_tmdb(tmdb_id):
    """
    Récupère les détails complets d'un film depuis TMDb
    
    Args:
        tmdb_id: ID TMDb du film
    
    Returns:
        dict avec toutes les infos
    """
    try:
        url = f"{TMDB_BASE_URL}/movie/{tmdb_id}"
        params = {
            "api_key": TMDB_API_KEY,
            "language": "fr-FR",
            "append_to_response": "credits"
        }
        
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            result = {
                'tmdb_id': tmdb_id,
                'title': data.get('title', ''),
                'original_title': data.get('original_title', ''),
                'synopsis': data.get('overview', 'Synopsis non disponible.'),
                'poster_path': data.get('poster_path'),
                'backdrop_path': data.get('backdrop_path'),
                'release_date': data.get('release_date'),
                'runtime': data.get('runtime'),
                'vote_average': data.get('vote_average'),
                'vote_count': data.get('vote_count'),
                'genres': [g['name'] for g in data.get('genres', [])],
                'director': 'Inconnu',
                'video': data.get('video'),
                'cast': []
            }
            
            # Directeur
            if 'credits' in data and 'crew' in data['credits']:
                directors = [
                    person['name'] 
                    for person in data['credits']['crew'] 
                    if person.get('job') == 'Director'
                ]
                result['director'] = directors[0] if directors else 'Inconnu'
            
            # Acteurs (5 premiers)
            if 'credits' in data and 'cast' in data['credits']:
                cast = data['credits']['cast'][:5]
                result['cast'] = [actor['name'] for actor in cast]
            
            # URLs images
            if result['poster_path']:
                result['poster_url'] = f"{TMDB_IMAGE_BASE}{result['poster_path']}"
            else:
                result['poster_url'] = f"https://via.placeholder.com/500x750/2F5233/FFFFFF?text={result['title']}"
            
            if result['backdrop_path']:
                result['backdrop_url'] = f"https://image.tmdb.org/t/p/original{result['backdrop_path']}"
            else:
                result['backdrop_url'] = None
            
            return result
        
        return None
        
    except Exception as e:
        print(f"Erreur détails TMDb : {e}")
        return None


@st.cache_data(ttl=3600)  # Cache 1h
def get_films_affiche_enrichis():
    """
    Récupère les films actuellement à l'affiche en France et les enrichit avec TMDb.
    Mode dégradé : utilise un cache statique si l'API n'est pas accessible.
    Retourne une liste de films avec toutes les infos (poster, synopsis, acteurs, etc.)
    """
    try:
        # Tenter de récupérer depuis l'API TMDb
        print("🔍 Tentative de récupération depuis API TMDb...")
        films_now_playing = get_now_playing_france()
        
        print(f"📊 Type retourné: {type(films_now_playing)}")
        print(f"📊 Nombre de films: {len(films_now_playing) if films_now_playing else 0}")
        
        # Si l'API a fonctionné
        if films_now_playing and len(films_now_playing) > 0:
            films_enrichis = []
            
            print(f"🔄 Enrichissement de {len(films_now_playing)} films...")
            
            for idx, film in enumerate(films_now_playing):
                # Récupérer détails complets depuis TMDb
                tmdb_id = film.get('id')
                if not tmdb_id:
                    print(f"⚠️ Film {idx}: pas de TMDb ID")
                    continue
                
                print(f"  Film {idx+1}/{len(films_now_playing)}: {film.get('title')} (ID: {tmdb_id})")
                
                details = get_movie_details_from_tmdb(tmdb_id)
                
                if details:
                    # Extraire l'année depuis release_date
                    annee = None
                    if film.get('release_date'):
                        try:
                            annee = int(film['release_date'][:4])
                        except:
                            pass
                    
                    # Combiner les infos
                    film_complet = {
                        'tmdb_id': tmdb_id,
                        'titre': details.get('title', film.get('title', 'Sans titre')),  # ← Titre FR prioritaire de TMDb
                        'titre_original': details.get('original_title', film.get('original_title', '')),
                        'poster_url': details['poster_url'],
                        'backdrop_url': details.get('backdrop_url'),
                        'synopsis': details['synopsis'],
                        'note': film.get('vote_average', 0),
                        'nb_votes': film.get('vote_count', 0),
                        'annee': annee,
                        'date_sortie': film.get('release_date', ''),
                        'realisateur': details.get('director', 'Inconnu'),
                        'acteurs': details.get('cast', []),
                        'genres': details.get('genres', []),
                        'duree': details.get('runtime'),
                        'langue_originale': film.get('original_language', ''),
                        'popularite': film.get('popularity', 0),
                    }
                    
                    films_enrichis.append(film_complet)
                else:
                    print(f"  ⚠️ Pas de détails pour {film.get('title')}")
            
            print(f"✅ {len(films_enrichis)} films enrichis avec succès (API)")
            return films_enrichis
        
        else:
            # Mode dégradé : utiliser le cache statique
            print("⚠️ API non accessible, utilisation du cache statique")
            try:
                from films_cache import FILMS_AFFICHE_CACHE
                print(f"✅ {len(FILMS_AFFICHE_CACHE)} films chargés depuis le cache")
                return FILMS_AFFICHE_CACHE
            except ImportError:
                print("❌ Cache statique non disponible")
                return []
    
    except Exception as e:
        print(f"❌ Erreur get_films_affiche_enrichis: {e}")
        import traceback
        traceback.print_exc()
        
        # Tentative de chargement du cache en dernier recours
        try:
            from films_cache import FILMS_AFFICHE_CACHE
            print(f"💾 Chargement du cache de secours ({len(FILMS_AFFICHE_CACHE)} films)")
            return FILMS_AFFICHE_CACHE
        except:
            return []


def enrich_movie_with_tmdb(movie_row):
    """
    Enrichit une ligne de DataFrame avec les infos TMDb
    
    Args:
        movie_row: Series (ligne du DataFrame)
    
    Returns:
        dict avec infos enrichies
    """
    # Récupérer titre et année
    title = movie_row.get('titre') or movie_row.get('primaryTitle')
    year = None
    
    if 'startYear' in movie_row and pd.notna(movie_row['startYear']):
        try:
            year = int(movie_row['startYear'])
        except:
            year = None
    
    # Recherche sur TMDb
    tmdb_result = search_tmdb_by_title(title, year)
    
    if tmdb_result:
        tmdb_id = tmdb_result['id']
        details = get_movie_details_from_tmdb(tmdb_id)
        
        if details:
            return {
                'tconst': movie_row.get('tconst'),
                'title': title,
                'year': year,
                'rating': movie_row.get('note') or movie_row.get('averageRating'),
                'votes': movie_row.get('votes') or movie_row.get('numVotes'),
                'runtime': details.get('runtime') or movie_row.get('durée') or movie_row.get('runtimeMinutes'),
                'genres': details.get('genres', []),
                'director': details.get('director', 'Inconnu'),
                'cast': details.get('cast', []),
                'synopsis': details.get('synopsis', 'Synopsis non disponible.'),
                'poster_url': details.get('poster_url'),
                'backdrop_url': details.get('backdrop_url'),
                'tmdb_id': tmdb_id
            }
    
    # Fallback si échec
    return {
        'tconst': movie_row.get('tconst'),
        'title': title,
        'year': year,
        'rating': movie_row.get('note') or movie_row.get('averageRating'),
        'votes': movie_row.get('votes') or movie_row.get('numVotes'),
        'runtime': movie_row.get('durée') or movie_row.get('runtimeMinutes'),
        'genres': [],
        'director': 'Inconnu',
        'cast': [],
        'synopsis': 'Synopsis non disponible.',
        'poster_url': f"https://via.placeholder.com/500x750/2F5233/FFFFFF?text={title}",
        'backdrop_url': None,
        'tmdb_id': None
    }


# ==========================================
# FONCTIONS GRAPHIQUES
# ==========================================

def create_styled_barplot(data, x, y, title, hue=None, palette=None, 
                          xlabel='', ylabel='', rotation=0, figsize=(10, 6),
                          show_values=True, value_format='%.1f%%'):
    """Crée un barplot stylisé"""
    fig, ax = plt.subplots(figsize=figsize)
    
    if palette is None:
        palette = PALETTE_CREUSE['gradient'] if hue is None else [PALETTE_CREUSE['bleu'], PALETTE_CREUSE['rouge']]
    
    sns.barplot(data=data, x=x, y=y, hue=hue, palette=palette, ax=ax, edgecolor='white', linewidth=1.5)
    
    if show_values:
        for container in ax.containers:
            ax.bar_label(container, fmt=value_format, padding=3, fontsize=9, fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold', color=PALETTE_CREUSE['principal'], pad=15)
    
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, fontweight='600', color=PALETTE_CREUSE['neutre'])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, fontweight='600', color=PALETTE_CREUSE['neutre'])
    
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(PALETTE_CREUSE['neutre'])
    ax.spines['bottom'].set_color(PALETTE_CREUSE['neutre'])
    
    if rotation > 0:
        plt.xticks(rotation=rotation, ha='right')
    
    ax.tick_params(colors=PALETTE_CREUSE['neutre'], labelsize=10)
    ax.set_facecolor('#F8F9FA')
    
    if hue is not None:
        legend = ax.legend(title_fontsize=11, fontsize=10, frameon=True, facecolor='white', 
                          edgecolor=PALETTE_CREUSE['neutre'], loc='best')
        if legend.get_title():
            legend.get_title().set_color(PALETTE_CREUSE['principal'])
    
    plt.tight_layout()
    return fig, ax


# ==========================================
# FONCTIONS UTILITAIRES
# ==========================================

def get_project_root():
    """Trouve la racine du projet"""
    return Path(__file__).resolve().parent.parent


def format_genre(genre):
    """Formate le genre pour affichage"""
    if isinstance(genre, list):
        return ', '.join(genre) if genre else 'Non spécifié'
    elif isinstance(genre, str):
        return genre.replace(',', ', ')
    return 'Non spécifié'


def safe_get(row, key, default='N/A'):
    """Récupère une valeur avec fallback"""
    try:
        val = row.get(key, default)
        return val if pd.notna(val) else default
    except:
        return default


def check_password():
    """Authentification pour gérants"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if st.session_state.authenticated:
        return True
    
    st.markdown("### 🔐 Accès Réservé aux Gérants")
    st.markdown("Cette page contient l'étude de marché complète et les données sensibles.")
    
    with st.form("login_form"):
        username = st.text_input("Identifiant")
        password = st.text_input("Mot de passe", type="password")
        submit = st.form_submit_button("Se connecter")
        
        if submit:
            if username in ADMIN_CREDENTIALS and ADMIN_CREDENTIALS[username] == password:
                st.session_state.authenticated = True
                st.session_state.authenticated_user = username  # ← AJOUT : Stocker l'utilisateur
                st.success(f"✅ Connexion réussie ! Bienvenue {username}")
                st.rerun()
            else:
                st.error("❌ Identifiant ou mot de passe incorrect")
    
    st.info("💡 **Identifiants** : `paul` / `WCS26`")
    return False


def create_map(user_location=None):
    """Crée une carte interactive avec les cinémas"""
    center_lat = 46.1
    center_lon = 1.9
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="OpenStreetMap")
    
    for cinema in CINEMAS:
        popup_html = f"""
        <div style='width: 200px'>
            <h4>{cinema['nom']}</h4>
            <p><b>{cinema['ville']}</b></p>
            <p>{cinema['adresse']}</p>
            <p>📞 {cinema['telephone']}</p>
        </div>
        """
        
        folium.Marker(
            location=[cinema['lat'], cinema['lon']],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=cinema['nom'],
            icon=folium.Icon(color='green', icon='film', prefix='fa')
        ).add_to(m)
    
    if user_location:
        folium.Marker(
            location=user_location,
            popup="Votre position",
            tooltip="Vous êtes ici",
            icon=folium.Icon(color='red', icon='user', prefix='fa')
        ).add_to(m)
        
        min_dist = float('inf')
        closest_cinema = None
        
        for cinema in CINEMAS:
            dist = ((cinema['lat'] - user_location[0])**2 + (cinema['lon'] - user_location[1])**2)**0.5
            if dist < min_dist:
                min_dist = dist
                closest_cinema = cinema
        
        if closest_cinema:
            folium.PolyLine(
                locations=[user_location, [closest_cinema['lat'], closest_cinema['lon']]],
                color='blue', weight=3, opacity=0.7,
                popup=f"Distance vers {closest_cinema['nom']}"
            ).add_to(m)
    
    return m


# ==========================================
# FONCTIONS POUR PAGE CINÉMAS
# ==========================================

@st.cache_data(ttl=86400)  # Cache 24h
def get_now_playing_france():
    """
    Récupère TOUS les films actuellement à l'affiche en France
    Pagination automatique pour récupérer toutes les pages
    
    Returns:
        pd.DataFrame: Films à l'affiche avec colonnes TMDb
    """
    now_playing_list = []
    j = 0
    
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJhODYxN2NkZDNiOTNmOGEzNTNmMjRhMTg0M2NjYWFmYiIsIm5iZiI6MTc2NTg4MzI0MS41MzEwMDAxLCJzdWIiOiI2OTQxM2Q2OTMyNzVjYjA1NWRjZmVkNDUiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.vnLVfgVtlhmQtbEp9BwvFMnL9u-J6CoCQVxP_bDYFQM"
    }
    
    while True:
        j += 1
        url = f"https://api.themoviedb.org/3/movie/now_playing?page={j}&region=FR"
        
        try:
            response_now = requests.get(url, headers=headers, timeout=10)
            
            if response_now.status_code == 200:
                data_now = response_now.json()
                now_playing = data_now['results']
                
                if len(now_playing) == 0:
                    break
                else:
                    now_playing_list.extend(now_playing)
            else:
                print(f"Erreur API page {j}: {response_now.status_code}")
                break
                
        except Exception as e:
            print(f"Erreur récupération page {j}: {e}")
            break
    
    print(f"✅ {len(now_playing_list)} films récupérés depuis TMDb")
    return now_playing_list  # Retourne la liste, pas un DataFrame


@st.cache_data(ttl=86400)
def match_now_playing_with_imdb(df_now_playing, df_imdb):
    """
    Match les films TMDb now_playing avec notre base IMDb
    
    Args:
        df_now_playing: DataFrame TMDb des films à l'affiche
        df_imdb: DataFrame IMDb complet
    
    Returns:
        list: Liste des films matchés avec leurs infos
    """
    matched_films = []
    
    for _, movie in df_now_playing.iterrows():
        title = movie.get('title', '')
        original_title = movie.get('original_title', '')
        release_year = None
        
        # Extraire l'année
        if 'release_date' in movie and pd.notna(movie['release_date']):
            try:
                release_year = int(movie['release_date'][:4])
            except:
                pass
        
        # Recherche dans IMDb
        # Stratégie 1 : Titre exact + année
        if release_year:
            matches = df_imdb[
                (df_imdb['titre'].str.lower() == title.lower()) &
                (df_imdb['startYear'] == release_year)
            ]
            
            if len(matches) > 0:
                matched_films.append({
                    'tconst': matches.iloc[0]['tconst'],
                    'tmdb_id': movie.get('id'),
                    'title': title,
                    'poster_path': movie.get('poster_path'),
                    'vote_average': movie.get('vote_average')
                })
                continue
        
        # Stratégie 2 : Titre contient (flexible)
        matches = df_imdb[
            df_imdb['titre'].str.contains(title, case=False, na=False, regex=False)
        ]
        
        if len(matches) > 0:
            # Prendre le mieux noté
            best_match = matches.nlargest(1, 'note').iloc[0]
            matched_films.append({
                'tconst': best_match['tconst'],
                'tmdb_id': movie.get('id'),
                'title': title,
                'poster_path': movie.get('poster_path'),
                'vote_average': movie.get('vote_average')
            })
            continue
        
        # Stratégie 3 : Titre original
        if original_title and original_title != title:
            matches = df_imdb[
                df_imdb['titre'].str.contains(original_title, case=False, na=False, regex=False)
            ]
            
            if len(matches) > 0:
                best_match = matches.nlargest(1, 'note').iloc[0]
                matched_films.append({
                    'tconst': best_match['tconst'],
                    'tmdb_id': movie.get('id'),
                    'title': original_title,
                    'poster_path': movie.get('poster_path'),
                    'vote_average': movie.get('vote_average')
                })
    
    return matched_films


def assign_films_to_cinemas(matched_films, cinemas, min_films=4, max_films=8):
    """
    Assigne des films à chaque cinéma de façon réaliste
    
    Args:
        matched_films: Liste des films matchés
        cinemas: Liste des cinémas
        min_films: Nombre minimum de films par cinéma
        max_films: Nombre maximum de films par cinéma
    
    Returns:
        dict: {nom_cinema: [film_dict1, film_dict2, ...]}
    """
    import random
    
    cinema_films = {}
    
    # Les grands cinémas (Guéret, La Souterraine) ont plus de films
    cinema_sizes = {
        "Cinéma Le Sénéchal": "grand",
        "Cinéma Eden": "grand",
        "Cinéma Colbert": "moyen",
        "Cinéma Claude Miller": "moyen",
        "Cinéma Alpha": "petit",
        "Cinéma Le Marchois": "petit",
        "Salle des Fêtes (Cinéma)": "petit"
    }
    
    for cinema in cinemas:
        cinema_name = cinema['nom']
        size = cinema_sizes.get(cinema_name, "moyen")
        
        # Adapter le nombre de films selon la taille
        if size == "grand":
            nb_films = random.randint(max_films - 2, max_films)
        elif size == "moyen":
            nb_films = random.randint(min_films + 1, max_films - 2)
        else:
            nb_films = random.randint(min_films, min_films + 2)
        
        # Sélectionner aléatoirement
        if len(matched_films) >= nb_films:
            selected = random.sample(matched_films, nb_films)
        else:
            selected = matched_films.copy()
        
        cinema_films[cinema_name] = selected
    
    return cinema_films


def assign_films_to_cinemas_enrichis(films_enrichis, cinemas, min_films=4, max_films=8):
    """
    Assigne des films enrichis à chaque cinéma de façon réaliste
    Version simplifiée qui prend directement les films enrichis
    
    Args:
        films_enrichis: Liste des films enrichis (avec toutes les infos)
        cinemas: Liste des cinémas
        min_films: Nombre minimum de films par cinéma
        max_films: Nombre maximum de films par cinéma
    
    Returns:
        dict: {nom_cinema: [film_dict1, film_dict2, ...]}
    """
    import random
    
    cinema_films = {}
    
    # Les grands cinémas (Guéret, La Souterraine) ont plus de films
    cinema_sizes = {
        "Cinéma Le Sénéchal": "grand",
        "Cinéma Eden": "grand",
        "Cinéma Colbert": "moyen",
        "Cinéma Claude Miller": "moyen",
        "Cinéma Alpha": "petit",
        "Cinéma Le Marchois": "petit",
        "Salle des Fêtes (Cinéma)": "petit"
    }
    
    for cinema in cinemas:
        cinema_name = cinema['nom']
        size = cinema_sizes.get(cinema_name, "moyen")
        
        # Adapter le nombre de films selon la taille
        if size == "grand":
            nb_films = random.randint(max_films - 2, max_films)
        elif size == "moyen":
            nb_films = random.randint(min_films + 1, max_films - 2)
        else:
            nb_films = random.randint(min_films, min_films + 2)
        
        # Limiter au nombre de films disponibles
        nb_films = min(nb_films, len(films_enrichis))
        
        # Sélectionner aléatoirement
        if len(films_enrichis) >= nb_films:
            selected = random.sample(films_enrichis, nb_films)
        else:
            selected = films_enrichis.copy()
        
        cinema_films[cinema_name] = selected
    
    return cinema_films


def calculate_cinema_distance(cinema, user_location):
    """
    Calcule la distance entre un cinéma et la position utilisateur
    
    Args:
        cinema: dict avec lat/lon
        user_location: [lat, lon]
    
    Returns:
        float: distance en km
    """
    if not user_location:
        return 0
    
    dist = ((cinema['lat'] - user_location[0])**2 + 
            (cinema['lon'] - user_location[1])**2)**0.5
    dist_km = dist * 111  # Conversion approximative en km
    
    return dist_km


# ==========================================
# RECHERCHE SIMPLE ET FIABLE (SANS IA)
# ==========================================

# Désactiver la recherche sémantique (trop instable et donne des résultats faux)
USE_SEMANTIC_SEARCH = False


def normalize_text(text):
    """
    Normalise un texte pour la recherche (supprime accents, apostrophes, caractères spéciaux)
    
    Args:
        text: Texte à normaliser
    
    Returns:
        str: Texte normalisé en minuscules sans accents
    """
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    
    # Supprimer les accents
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    
    # Remplacer apostrophes et caractères spéciaux par espaces
    special_chars = ["'", "'", "`", "-", "_", ":", ";", ",", ".", "!", "?"]
    for char in special_chars:
        text = text.replace(char, " ")
    
    # Supprimer espaces multiples
    text = " ".join(text.split())
    
    return text


def simple_similarity(str1, str2):
    """
    Calcule la similarité entre deux chaînes avec SequenceMatcher
    
    Args:
        str1: Première chaîne
        str2: Deuxième chaîne
    
    Returns:
        float: Score entre 0 et 1
    """
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()


def fuzzy_match_score(query_norm, title_norm, query_words):
    """
    Calcule un score de correspondance entre une requête et un titre (normalisés)
    
    Args:
        query_norm: Requête normalisée
        title_norm: Titre normalisé
        query_words: Liste des mots de la requête
    
    Returns:
        float: Score entre 0 et 100
    """
    if not query_norm:
        return 0
    
    # 1. Correspondance exacte = 100 points
    if query_norm == title_norm:
        return 100
    
    # 2. Titre commence par la requête = 90 points
    if title_norm.startswith(query_norm):
        return 90
    
    # 3. Requête contenue dans le titre = 80 points
    if query_norm in title_norm:
        return 80
    
    # 4. Tous les mots de la requête sont dans le titre = 70 points
    if len(query_words) > 0 and all(word in title_norm for word in query_words if len(word) >= 2):
        return 70
    
    # 5. Au moins 50% des mots correspondent = 50-60 points
    if len(query_words) > 0:
        matching_words = sum(1 for word in query_words if len(word) >= 2 and word in title_norm)
        if matching_words > 0:
            return 40 + (matching_words / len(query_words)) * 30
    
    # 6. Similarité de base avec SequenceMatcher = 0-40 points
    sim = simple_similarity(query_norm, title_norm)
    return sim * 40


def find_movies_with_correction(query, df, max_results=10, prefer_french=True):
    """
    Recherche de films OPTIMISÉE avec support bilingue (FR/EN)
    Priorise les résultats en français quand disponible
    
    Args:
        query: requête de recherche
        df: DataFrame contenant les films
        max_results: nombre maximum de résultats
        prefer_french: Si True, priorise les matchs sur titres français
    
    Returns:
        tuple: (DataFrame des résultats, correction suggérée ou None, message d'info)
    """
    if not query or len(query.strip()) < 2:
        return pd.DataFrame(), None, None
    
    query = query.strip()
    query_norm = normalize_text(query)
    query_words = query_norm.split()
    
    # ==========================================
    # PRÉPARATION : Index de recherche optimisé
    # ==========================================
    
    df_work = df.copy()
    
    # Colonnes de titres possibles (ordre de priorité)
    title_columns_priority = [
        'frenchTitle',           # 1. Titre français depuis IMDb akas
        'titre_francais',        # 2. Titre français alternatif
        'titre',                 # 3. Titre principal
        'primaryTitle',          # 4. Titre IMDb principal
        'originalTitle',         # 5. Titre original
        'localizedTitle'         # 6. Titre localisé TMDb
    ]
    
    # Identifier les colonnes disponibles
    available_columns = [col for col in title_columns_priority if col in df_work.columns]
    
    if not available_columns:
        return pd.DataFrame(), None, "❌ Aucune colonne de titre trouvée"
    
    # Créer deux colonnes : une pour FR, une pour toutes
    if prefer_french:
        # Priorité française : frenchTitle > titre_francais > titre
        french_cols = [col for col in ['frenchTitle', 'titre_francais', 'titre'] if col in available_columns]
        if french_cols:
            df_work['search_primary'] = df_work[french_cols].fillna('').apply(
                lambda row: next((str(val) for val in row if val), ''),
                axis=1
            )
        else:
            df_work['search_primary'] = df_work[available_columns[0]]
        
        # Tous les titres pour fallback
        df_work['search_all'] = df_work[available_columns].fillna('').apply(
            lambda row: ' | '.join([str(val) for val in row if val]),
            axis=1
        )
    else:
        # Sans priorité : chercher dans tous les titres
        df_work['search_primary'] = df_work[available_columns].fillna('').apply(
            lambda row: ' | '.join([str(val) for val in row if val]),
            axis=1
        )
        df_work['search_all'] = df_work['search_primary']
    
    # Normaliser
    df_work['primary_norm'] = df_work['search_primary'].apply(normalize_text)
    df_work['all_norm'] = df_work['search_all'].apply(normalize_text)
    
    # ==========================================
    # ÉTAPE 1 : Recherche EXACTE sur titres prioritaires
    # ==========================================
    
    exact_matches = df_work[
        df_work['primary_norm'].str.contains(f'\\b{query_norm}\\b', na=False, regex=True)
    ]
    
    if len(exact_matches) > 0:
        result = exact_matches.drop(
            ['search_primary', 'search_all', 'primary_norm', 'all_norm'], 
            axis=1, errors='ignore'
        ).head(max_results)
        return result, None, f"✅ {len(exact_matches)} résultat(s) exact(s) trouvé(s)"
    
    # ==========================================
    # ÉTAPE 2 : Recherche "CONTIENT" sur titres prioritaires
    # ==========================================
    
    contains_matches = df_work[
        df_work['primary_norm'].str.contains(query_norm, na=False, regex=False)
    ]
    
    if len(contains_matches) > 0:
        result = contains_matches.drop(
            ['search_primary', 'search_all', 'primary_norm', 'all_norm'], 
            axis=1, errors='ignore'
        ).head(max_results)
        return result, None, f"✅ {len(contains_matches)} résultat(s) trouvé(s)"
    
    # ==========================================
    # ÉTAPE 3 : Recherche sur TOUS les titres (fallback)
    # ==========================================
    
    all_matches = df_work[
        df_work['all_norm'].str.contains(query_norm, na=False, regex=False)
    ]
    
    if len(all_matches) > 0:
        result = all_matches.drop(
            ['search_primary', 'search_all', 'primary_norm', 'all_norm'], 
            axis=1, errors='ignore'
        ).head(max_results)
        return result, None, f"✅ {len(all_matches)} résultat(s) trouvé(s) (titre original)"
    
    # ==========================================
    # ÉTAPE 4 : Recherche par MOTS MULTIPLES
    # ==========================================
    
    if len(query_words) > 1:
        mask = pd.Series([True] * len(df_work))
        for word in query_words:
            if len(word) >= 2:
                mask &= (
                    df_work['primary_norm'].str.contains(word, na=False, regex=False) |
                    df_work['all_norm'].str.contains(word, na=False, regex=False)
                )
        
        word_matches = df_work[mask]
        
        if len(word_matches) > 0:
            result = word_matches.drop(
                ['search_primary', 'search_all', 'primary_norm', 'all_norm'], 
                axis=1, errors='ignore'
            ).head(max_results)
            return result, None, f"💡 {len(word_matches)} résultat(s) trouvé(s) par mots-clés"
    
    # ==========================================
    # ÉTAPE 5 : Recherche FLOUE avec score
    # ==========================================
    
    scores = []
    sample_size = min(15000, len(df_work))
    
    for idx in range(sample_size):
        row = df_work.iloc[idx]
        
        # Score sur titre prioritaire (poids 70%)
        score_primary = fuzzy_match_score(query_norm, row['primary_norm'], query_words) * 0.7
        
        # Score sur tous les titres (poids 30%)
        score_all = fuzzy_match_score(query_norm, row['all_norm'], query_words) * 0.3
        
        total_score = score_primary + score_all
        
        if total_score >= 25:  # Seuil minimum abaissé
            title_display = get_display_title(df.iloc[row.name], prefer_french=prefer_french, include_year=False)
            scores.append((row.name, total_score, title_display))
    
    # Trier par score décroissant
    scores.sort(key=lambda x: x[1], reverse=True)
    
    if len(scores) > 0:
        top_indices = [idx for idx, score, title in scores[:max_results]]
        result = df.loc[top_indices]
        
        best_idx, best_score, best_title = scores[0]
        
        if best_score < 100 and best_score >= 40:
            message = f"💡 Meilleur résultat : **{best_title}** (confiance: {int(best_score)}%)"
            return result, best_title, message
        else:
            return result, None, f"✅ {len(scores)} résultat(s) trouvé(s) (recherche approchante)"
    
    # ==========================================
    # ÉTAPE 6 : Aucun résultat
    # ==========================================
    
    return pd.DataFrame(), None, f"❌ Aucun film trouvé pour '{query}'"


# ==========================================
# DIAGNOSTIC DES COLONNES DE TITRES
# ==========================================

def check_title_columns(df):
    """
    Vérifie quelles colonnes de titres sont disponibles dans le DataFrame
    et teste si des titres français sont présents
    
    Args:
        df: DataFrame IMDb
    
    Returns:
        dict: Informations sur les colonnes de titres
    """
    results = {
        'all_columns': df.columns.tolist(),
        'title_columns': [],
        'has_french_titles': False,
        'french_test_results': {},
        'samples': {},
        'recommendations': []
    }
    
    # Trouver les colonnes avec "title" ou "titre"
    title_cols = [col for col in df.columns if 'title' in col.lower() or 'titre' in col.lower()]
    results['title_columns'] = title_cols
    
    # Tester la recherche de films français typiques
    french_queries = ['Bienvenue', 'Intouchables', 'Amélie']
    
    for query in french_queries:
        results['french_test_results'][query] = {}
        
        for col in title_cols:
            try:
                matches = df[df[col].str.contains(query, case=False, na=False)]
                results['french_test_results'][query][col] = {
                    'count': len(matches),
                    'example': matches[col].iloc[0] if len(matches) > 0 else None
                }
                
                if len(matches) > 0:
                    results['has_french_titles'] = True
            except:
                results['french_test_results'][query][col] = {
                    'count': 0,
                    'example': None,
                    'error': True
                }
    
    # Échantillon de films pour chaque colonne
    for col in title_cols[:3]:  # Max 3 colonnes
        try:
            results['samples'][col] = df[col].head(5).tolist()
        except:
            results['samples'][col] = []
    
    # Recommandations
    if 'titre_francais' in title_cols or 'frenchTitle' in title_cols:
        results['recommendations'].append({
            'type': 'success',
            'message': "✅ Colonne de titres français détectée"
        })
    elif any('localized' in col.lower() for col in title_cols):
        results['recommendations'].append({
            'type': 'warning',
            'message': "⚠️ Colonne 'localized' détectée - vérifiez si elle contient des titres français"
        })
    else:
        results['recommendations'].append({
            'type': 'error',
            'message': "❌ Aucune colonne de titres français détectée"
        })
        results['recommendations'].append({
            'type': 'info',
            'message': "💡 Ajoutez la table IMDb akas pour les titres alternatifs"
        })
    
    return results


# ==========================================
# AFFICHAGE VIDÉO YOUTUBE RESPONSIVE
# ==========================================

def display_youtube_video(video_id, title="", director="", max_width=800):
    """
    Affiche une vidéo YouTube de manière responsive avec iframe HTML
    
    Args:
        video_id: ID de la vidéo YouTube (ex: 'd9MyW72ELq0')
        title: Titre du film (optionnel)
        director: Nom du réalisateur (optionnel)
        max_width: Largeur maximale en pixels (défaut: 800)
    
    Example:
        display_youtube_video(
            video_id="d9MyW72ELq0",
            title="Avatar: The Way of Water",
            director="James Cameron"
        )
    """
    video_html = f"""
    <div style="max-width: {max_width}px; margin: 0 auto;">
        <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden;">
            <iframe 
                src="https://www.youtube.com/embed/{video_id}" 
                style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"
                frameborder="0" 
                allowfullscreen>
            </iframe>
        </div>
    </div>
    """
    
    st.markdown(video_html, unsafe_allow_html=True)
    
    # Afficher le titre et le réalisateur si fournis
    if title and director:
        st.caption(f"🎬 {title} - {director}")
    elif title:
        st.caption(f"🎬 {title}")


def get_movie_trailer(tmdb_id):
    """
    Récupère l'URL du trailer YouTube depuis l'API TMDb
    
    Args:
        tmdb_id: ID TMDb du film
    
    Returns:
        str: ID YouTube de la vidéo (ex: 'd9MyW72ELq0') ou None si pas de trailer
    """
    try:
        url = f"{TMDB_BASE_URL}/movie/{tmdb_id}/videos"
        params = {
            'api_key': TMDB_API_KEY,
            'language': 'fr-FR'
        }
        
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            videos = data.get('results', [])
            
            # Chercher d'abord un trailer en français
            for video in videos:
                if (video.get('type') == 'Trailer' and 
                    video.get('site') == 'YouTube' and
                    video.get('iso_639_1') == 'fr'):
                    return video.get('key')
            
            # Si pas de trailer français, prendre un trailer anglais
            for video in videos:
                if (video.get('type') == 'Trailer' and 
                    video.get('site') == 'YouTube'):
                    return video.get('key')
        
        return None
        
    except Exception as e:
        print(f"Erreur récupération trailer pour film {tmdb_id}: {e}")
        return None


def get_trailers_from_films(films_list, max_trailers=10):
    """
    Récupère les trailers disponibles pour une liste de films
    
    Args:
        films_list: Liste de dictionnaires de films (avec tmdb_id)
        max_trailers: Nombre maximum de trailers à récupérer
    
    Returns:
        dict: Dictionnaire des trailers disponibles
              Format: {clé: {'video_id': str, 'titre': str, 'realisateur': str, 'film_data': dict}}
    """
    trailers_disponibles = {}
    count = 0
    
    for film in films_list:
        if count >= max_trailers:
            break
        
        tmdb_id = film.get('tmdb_id')
        if not tmdb_id:
            continue
        
        # Récupérer le trailer
        video_id = get_movie_trailer(tmdb_id)
        
        if video_id:
            # Créer une clé unique pour le film
            key = film.get('titre', f'Film_{tmdb_id}')
            
            trailers_disponibles[key] = {
                'video_id': video_id,
                'titre': film.get('titre', 'Sans titre'),
                'realisateur': film.get('realisateur', 'Réalisateur inconnu'),
                'film_data': film  # Garder toutes les données du film
            }
            count += 1
    
    return trailers_disponibles


# ==========================================
# RECOMMANDATIONS PERSONNALISÉES
# ==========================================

def calculate_film_similarity_score(film, liked_genres, disliked_film_ids):
    """
    Calcule un score de similarité pour un film basé sur les préférences utilisateur
    
    Args:
        film: DataFrame row du film
        liked_genres: Liste des genres préférés de l'utilisateur
        disliked_film_ids: Liste des IDs de films pas aimés (à exclure)
    
    Returns:
        float: Score de similarité (0-100)
    """
    film_id = film.get('tconst')
    
    # Exclure les films pas aimés
    if film_id and str(film_id) in disliked_film_ids:
        return 0
    
    score = 0
    
    # Genres (poids le plus important : 60 points max)
    film_genres = film.get('genres', '')
    if pd.notna(film_genres) and isinstance(film_genres, str):
        film_genres_list = [g.strip() for g in film_genres.split(',')]
        
        # Compter combien de genres préférés sont présents
        matching_genres = sum(1 for genre in liked_genres if genre in film_genres_list)
        
        if len(liked_genres) > 0:
            genre_score = (matching_genres / len(liked_genres)) * 60
            score += genre_score
    
    # Note IMDb (poids moyen : 30 points max)
    note = film.get('note', 0)
    if pd.notna(note) and note > 0:
        # Normaliser la note (films > 7/10 ont un bon score)
        note_score = ((note - 5) / 5) * 30 if note > 5 else 0
        score += max(0, note_score)
    
    # Popularité (votes IMDb) (poids faible : 10 points max)
    votes = film.get('votes', 0)
    if pd.notna(votes) and votes > 0:
        # Normaliser avec log (films avec beaucoup de votes)
        popularity_score = min(10, np.log10(votes + 1) * 2)
        score += popularity_score
    
    return min(100, score)


def get_personalized_recommendations(df_movies, liked_films, disliked_films, top_n=20):
    """
    Génère des recommandations personnalisées basées sur les films aimés
    
    Args:
        df_movies: DataFrame de tous les films disponibles
        liked_films: Liste de tuples (film_id, film_data) des films aimés
        disliked_films: Liste de tuples (film_id, film_data) des films pas aimés
        top_n: Nombre de recommandations à retourner
    
    Returns:
        DataFrame: Films recommandés avec scores
    """
    # Si aucun film aimé, retourner les films populaires
    if len(liked_films) == 0:
        # Retourner les films les mieux notés avec beaucoup de votes
        popular = df_movies[
            (df_movies['note'] >= 7.0) & 
            (df_movies['votes'] >= 50000)
        ].copy()
        
        popular['score_popularite'] = popular['note'] * np.log10(popular['votes'] + 1)
        popular = popular.sort_values('score_popularite', ascending=False)
        
        return popular.head(top_n)
    
    # Extraire les genres préférés
    liked_genres = []
    for _, film_data in liked_films:
        genres = film_data.get('genres', [])
        if isinstance(genres, list):
            liked_genres.extend(genres)
        elif isinstance(genres, str):
            liked_genres.extend([g.strip() for g in genres.split(',')])
    
    # Compter les occurrences et garder les plus fréquents
    from collections import Counter
    genre_counts = Counter(liked_genres)
    top_genres = [genre for genre, count in genre_counts.most_common(5)]
    
    # IDs des films déjà vus (aimés ou pas aimés) à exclure
    watched_ids = set()
    for film_id, _ in liked_films:
        watched_ids.add(str(film_id))
    
    disliked_ids = set()
    for film_id, _ in disliked_films:
        watched_ids.add(str(film_id))
        disliked_ids.add(str(film_id))
    
    # Calculer le score pour chaque film
    recommendations = []
    
    for idx, film in df_movies.iterrows():
        film_id = str(film.get('tconst', ''))
        
        # Exclure les films déjà vus
        if film_id in watched_ids:
            continue
        
        # Calculer le score de similarité
        similarity_score = calculate_film_similarity_score(film, top_genres, disliked_ids)
        
        if similarity_score > 30:  # Seuil minimum
            recommendations.append({
                'film': film,
                'score': similarity_score
            })
    
    # Trier par score
    recommendations.sort(key=lambda x: x['score'], reverse=True)
    
    # Prendre les top N
    top_recommendations = recommendations[:top_n]
    
    # Convertir en DataFrame
    if len(top_recommendations) > 0:
        films_data = [rec['film'] for rec in top_recommendations]
        scores = [rec['score'] for rec in top_recommendations]
        
        result_df = pd.DataFrame(films_data)
        result_df['score_recommandation'] = scores
        
        return result_df
    
    # Si aucune recommandation, retourner les populaires
    return get_personalized_recommendations(df_movies, [], disliked_films, top_n)

