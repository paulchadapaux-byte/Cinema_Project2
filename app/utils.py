"""
Fonctions utilitaires pour l'application Streamlit Cinéma Creuse
Inclut les appels API TMDb pour enrichissement des films
"""

import json
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
from datetime import datetime

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


def separer_films_par_statut(films_list):
    """
    Sépare les films entre ceux déjà en salles et ceux à venir
    basé sur la date de sortie vs date actuelle (dynamique)
    
    Args:
        films_list: Liste de films avec date_sortie
        
    Returns:
        tuple: (films_en_salles, films_a_venir)
    """
    from datetime import datetime
    
    en_salles = []
    a_venir = []
    date_reference = datetime.now()
    
    for film in films_list:
        try:
            date_sortie_str = film.get('date_sortie', '')
            if date_sortie_str:
                date_sortie = datetime.strptime(date_sortie_str, '%Y-%m-%d')
                
                if date_sortie <= date_reference:
                    en_salles.append(film)
                else:
                    a_venir.append(film)
            else:
                # Si pas de date, considérer comme en salles
                en_salles.append(film)
        except:
            # En cas d'erreur, mettre en salles
            en_salles.append(film)
    
    return en_salles, a_venir


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
    Génère des recommandations personnalisées basées sur KNN (K-Nearest Neighbors)
    
    Args:
        df_movies: DataFrame de tous les films disponibles
        liked_films: Liste de tuples (film_id, film_data) des films aimés
        disliked_films: Liste de tuples (film_id, film_data) des films pas aimés
        top_n: Nombre de recommandations à retourner
    
    Returns:
        DataFrame: Films recommandés avec scores
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import MultiLabelBinarizer
    import numpy as np
    
    # Si aucun film aimé, retourner les films populaires
    if len(liked_films) == 0:
        popular = df_movies[
            (df_movies['note'] >= 7.0) & 
            (df_movies['votes'] >= 50000)
        ].copy()
        
        popular['score_popularite'] = popular['note'] * np.log10(popular['votes'] + 1)
        popular = popular.sort_values('score_popularite', ascending=False)
        popular['score_recommandation'] = 80.0
        
        return popular.head(top_n)
    
    # Extraire les IDs des films aimés
    liked_ids = set([str(film_id) for film_id, _ in liked_films])
    disliked_ids = set([str(film_id) for film_id, _ in disliked_films])
    watched_ids = liked_ids.union(disliked_ids)
    
    # Préparer les features pour KNN
    # 1. Encoder les genres avec MultiLabelBinarizer
    all_genres = []
    for idx, row in df_movies.iterrows():
        genres = row.get('genre', [])
        if isinstance(genres, list) and len(genres) > 0:
            all_genres.append(genres)
        else:
            all_genres.append([])
    
    mlb = MultiLabelBinarizer()
    genre_features = mlb.fit_transform(all_genres)
    
    # 2. Ajouter la note normalisée (0-1)
    notes = df_movies['note'].fillna(0).values / 10.0
    notes = notes.reshape(-1, 1)
    
    # 3. Combiner les features
    features = np.hstack([genre_features, notes])
    
    # Créer le modèle KNN
    knn = NearestNeighbors(n_neighbors=min(50, len(df_movies)), metric='cosine')
    knn.fit(features)
    
    # Trouver les indices des films aimés dans le DataFrame
    liked_indices = []
    for film_id, _ in liked_films:
        matches = df_movies[df_movies['tconst'] == film_id]
        if len(matches) > 0:
            liked_indices.append(matches.index[0])
    
    if len(liked_indices) == 0:
        # Fallback sur méthode populaire
        return get_personalized_recommendations(df_movies, [], disliked_films, top_n)
    
    # Pour chaque film aimé, trouver les voisins les plus proches
    all_neighbors = set()
    all_scores = {}
    
    for idx in liked_indices:
        # Trouver les K voisins les plus proches
        distances, indices = knn.kneighbors([features[idx]], n_neighbors=min(30, len(df_movies)))
        
        for dist, neighbor_idx in zip(distances[0], indices[0]):
            if neighbor_idx == idx:  # Skip le film lui-même
                continue
            
            film_id = str(df_movies.iloc[neighbor_idx]['tconst'])
            
            # Exclure les films déjà vus
            if film_id in watched_ids:
                continue
            
            # Calculer le score (similarité cosine -> score 0-100)
            # Distance cosine : 0 = identique, 2 = opposé
            similarity = max(0, 1 - dist)  # Convertir distance en similarité
            score = similarity * 100
            
            # Si le film apparaît plusieurs fois, prendre le meilleur score
            if neighbor_idx not in all_scores or all_scores[neighbor_idx] < score:
                all_scores[neighbor_idx] = score
                all_neighbors.add(neighbor_idx)
    
    # Convertir en liste et trier par score
    recommendations = [(idx, all_scores[idx]) for idx in all_neighbors]
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    # Prendre les top N
    top_recommendations = recommendations[:top_n]
    
    if len(top_recommendations) > 0:
        # Créer le DataFrame de résultats
        result_indices = [idx for idx, score in top_recommendations]
        result_df = df_movies.iloc[result_indices].copy()
        result_df['score_recommandation'] = [score for idx, score in top_recommendations]
        
        return result_df
    
    # Si aucune recommandation, retourner les populaires
    return get_personalized_recommendations(df_movies, [], disliked_films, top_n)


# ==========================================
# RECHERCHE PAR ACTEUR/RÉALISATEUR
# ==========================================

def find_movies_by_actor(query, df, max_results=20):
    """
    Recherche des films par nom d'acteur ou réalisateur
    
    Args:
        query: Nom de l'acteur/réalisateur à chercher
        df: DataFrame des films
        max_results: Nombre maximum de résultats
    
    Returns:
        tuple: (DataFrame résultats, message)
    """
    if not query or len(query) < 2:
        return pd.DataFrame(), ""
    
    # Vérifier si les colonnes acteurs/réalisateurs existent
    has_acteurs = 'acteurs' in df.columns
    has_realisateurs = 'realisateurs' in df.columns
    
    if not has_acteurs and not has_realisateurs:
        message = (
            "⚠️ La recherche par acteur n'est pas disponible.\n\n"
            "Les colonnes 'acteurs' et 'realisateurs' n'existent pas dans le dataset."
        )
        return pd.DataFrame(), message
    
    query_norm = normalize_text(query.lower())
    results = []
    
    for idx, row in df.iterrows():
        found = False
        
        # Chercher dans les acteurs si la colonne existe
        if has_acteurs and not found:
            acteurs = row.get('acteurs')
            
            # Gérer différents formats
            if acteurs is not None:
                # Si c'est un numpy array ou une liste (itérable)
                if hasattr(acteurs, '__iter__') and not isinstance(acteurs, str):
                    try:
                        for acteur in acteurs:
                            if acteur and isinstance(acteur, str) and acteur.strip():
                                acteur_norm = normalize_text(acteur.lower())
                                if query_norm in acteur_norm:
                                    results.append(idx)
                                    found = True
                                    break
                    except:
                        pass
                
                # Si c'est une string (ex: "Brad Pitt, Edward Norton")
                elif isinstance(acteurs, str) and acteurs.strip():
                    acteurs_norm = normalize_text(acteurs.lower())
                    if query_norm in acteurs_norm:
                        results.append(idx)
                        found = True
        
        # Chercher dans les réalisateurs si la colonne existe
        if has_realisateurs and not found:
            realisateurs = row.get('realisateurs')
            
            # Gérer différents formats
            if realisateurs is not None:
                # Si c'est un numpy array ou une liste (itérable)
                if hasattr(realisateurs, '__iter__') and not isinstance(realisateurs, str):
                    try:
                        for real in realisateurs:
                            if real and isinstance(real, str) and real.strip():
                                real_norm = normalize_text(real.lower())
                                if query_norm in real_norm:
                                    results.append(idx)
                                    found = True
                                    break
                    except:
                        pass
                
                # Si c'est une string
                elif isinstance(realisateurs, str) and realisateurs.strip():
                    real_norm = normalize_text(realisateurs.lower())
                    if query_norm in real_norm:
                        results.append(idx)
                        found = True
    
    if len(results) == 0:
        return pd.DataFrame(), f"❌ Aucun film trouvé avec '{query}'"
    
    # Récupérer les films correspondants
    matching_df = df.loc[results].copy()
    
    # Trier par note décroissante
    if 'note' in matching_df.columns:
        matching_df = matching_df.sort_values('note', ascending=False, na_position='last')
    elif 'averageRating' in matching_df.columns:
        matching_df = matching_df.sort_values('averageRating', ascending=False, na_position='last')
    
    # Limiter les résultats
    matching_df = matching_df.head(max_results)
    
    message = f"✅ {len(matching_df)} film(s) trouvé(s) avec '{query}'"
    
    return matching_df, message


def find_movies_combined(query, df, max_results=15, search_type='all', prefer_french=True):
    """
    Recherche combinée par titre ET/OU acteur
    
    Args:
        query: Requête de recherche
        df: DataFrame des films
        max_results: Nombre maximum de résultats
        search_type: 'title', 'actor', ou 'all'
        prefer_french: Priorité aux titres français
    
    Returns:
        tuple: (DataFrame résultats, message)
    """
    if not query or len(query) < 2:
        return pd.DataFrame(), ""
    
    results_title = pd.DataFrame()
    results_actor = pd.DataFrame()
    message = ""
    
    # Recherche par titre
    if search_type in ['title', 'all']:
        results_title, _, msg_title = find_movies_with_correction(
            query, df, max_results=max_results, prefer_french=prefer_french
        )
        if len(results_title) > 0:
            message = msg_title
    
    # Recherche par acteur
    if search_type in ['actor', 'all']:
        results_actor, msg_actor = find_movies_by_actor(query, df, max_results=max_results)
        if len(results_actor) > 0:
            if message:
                message += f" | {msg_actor}"
            else:
                message = msg_actor
    
    # Combiner les résultats
    if search_type == 'title':
        combined = results_title
    elif search_type == 'actor':
        combined = results_actor
    else:  # 'all'
        # Concaténer et supprimer les doublons
        combined = pd.concat([results_title, results_actor]).drop_duplicates(subset=['tconst'])
        combined = combined.head(max_results)
    
    if len(combined) == 0:
        return pd.DataFrame(), f"❌ Aucun résultat pour '{query}'"
    
    return combined, message


# ==========================================
# GESTIONNAIRE DE PROFILS UTILISATEURS
# ==========================================

class UserManager:
    """Gère les profils utilisateurs et leurs préférences de films"""
    
    def __init__(self, data_dir="data/user_profiles"):
        """
        Initialise le gestionnaire de profils
        
        Args:
            data_dir: Répertoire où stocker les profils JSON
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_profile_path(self, username):
        """Retourne le chemin du fichier de profil pour un utilisateur"""
        return self.data_dir / f"{username}.json"
    
    def load_profile(self, username):
        """
        Charge le profil d'un utilisateur (ou crée un profil vide)
        
        Args:
            username: Nom d'utilisateur
        
        Returns:
            dict: Profil utilisateur
        """
        profile_path = self._get_profile_path(username)
        
        if profile_path.exists():
            with open(profile_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Créer un profil vide
            return {
                "username": username,
                "films_vus": {},
                "date_creation": datetime.now().isoformat()
            }
    
    def save_profile(self, username, profile_data):
        """
        Sauvegarde le profil d'un utilisateur
        
        Args:
            username: Nom d'utilisateur
            profile_data: Données du profil
        """
        profile_path = self._get_profile_path(username)
        
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(profile_data, f, indent=2, ensure_ascii=False)
    
    def add_film(self, username, film_data, rating):
        """
        Ajoute un film au profil de l'utilisateur
        
        Args:
            username: Nom d'utilisateur
            film_data: Dictionnaire avec les infos du film
            rating: 'liked' ou 'disliked'
        
        Returns:
            bool: True si succès
        """
        profile = self.load_profile(username)
        
        # Récupérer l'ID du film
        film_id = film_data.get('tconst') or film_data.get('imdb_id') or film_data.get('id')
        
        if not film_id:
            return False
        
        # Ajouter le film
        profile['films_vus'][str(film_id)] = {
            "titre": film_data.get('titre', 'Titre inconnu'),
            "annee": film_data.get('startYear') or film_data.get('annee'),
            "rating": rating,
            "date_ajout": datetime.now().isoformat(),
            "genres": film_data.get('genre', []),  # Liste de genres
            "note": film_data.get('note'),
            "votes": film_data.get('votes')
        }
        
        self.save_profile(username, profile)
        return True
    
    def update_film_rating(self, username, film_id, new_rating):
        """
        Modifie la note d'un film
        
        Args:
            username: Nom d'utilisateur
            film_id: ID du film
            new_rating: 'liked' ou 'disliked'
        
        Returns:
            bool: True si succès
        """
        profile = self.load_profile(username)
        
        if str(film_id) in profile['films_vus']:
            profile['films_vus'][str(film_id)]['rating'] = new_rating
            profile['films_vus'][str(film_id)]['date_modification'] = datetime.now().isoformat()
            self.save_profile(username, profile)
            return True
        
        return False
    
    def remove_film(self, username, film_id):
        """
        Supprime un film du profil
        
        Args:
            username: Nom d'utilisateur
            film_id: ID du film
        
        Returns:
            bool: True si succès
        """
        profile = self.load_profile(username)
        
        if str(film_id) in profile['films_vus']:
            del profile['films_vus'][str(film_id)]
            self.save_profile(username, profile)
            return True
        
        return False
    
    def get_liked_films(self, username):
        """Récupère la liste des films aimés"""
        profile = self.load_profile(username)
        
        liked = [
            (film_id, film_data) 
            for film_id, film_data in profile['films_vus'].items() 
            if film_data.get('rating') == 'liked'
        ]
        
        return liked
    
    def get_disliked_films(self, username):
        """Récupère la liste des films pas aimés"""
        profile = self.load_profile(username)
        
        disliked = [
            (film_id, film_data) 
            for film_id, film_data in profile['films_vus'].items() 
            if film_data.get('rating') == 'disliked'
        ]
        
        return disliked
    
    def get_statistics(self, username):
        """
        Calcule des statistiques sur les films vus
        
        Args:
            username: Nom d'utilisateur
        
        Returns:
            dict: Statistiques
        """
        profile = self.load_profile(username)
        films_vus = profile['films_vus']
        
        nb_total = len(films_vus)
        nb_liked = sum(1 for f in films_vus.values() if f.get('rating') == 'liked')
        nb_disliked = sum(1 for f in films_vus.values() if f.get('rating') == 'disliked')
        
        # Genres préférés (des films aimés)
        genres_count = {}
        for film in films_vus.values():
            if film.get('rating') == 'liked' and film.get('genres'):
                genres = film['genres']
                
                # Gérer les deux formats : liste ou string
                if isinstance(genres, list):
                    genre_list = genres
                elif isinstance(genres, str):
                    genre_list = [g.strip() for g in genres.split(',') if g.strip()]
                else:
                    genre_list = []
                
                # Compter chaque genre
                for genre in genre_list:
                    if genre:
                        genres_count[genre] = genres_count.get(genre, 0) + 1
        
        genres_preferes = sorted(genres_count.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "nb_total": nb_total,
            "nb_liked": nb_liked,
            "nb_disliked": nb_disliked,
            "pourcentage_liked": (nb_liked / nb_total * 100) if nb_total > 0 else 0,
            "genres_preferes": genres_preferes
        }
    
    def is_film_already_rated(self, username, film_id):
        """Vérifie si un film a déjà été noté"""
        profile = self.load_profile(username)
        
        if str(film_id) in profile['films_vus']:
            return profile['films_vus'][str(film_id)].get('rating')
        
        return None
    
    def get_all_films(self, username):
        """Récupère tous les films vus"""
        profile = self.load_profile(username)
        return profile['films_vus']


# ==========================================
# CACHE PROFIL PAUL
# ==========================================

PAUL_LIKED_FILMS = {
    # Films de Guerre (9)
    "tt0056119": {"titre": "Le Jour le plus long", "annee": 1962, "rating": "liked", "genres": ["War", "Drama", "History"], "note": 7.7, "votes": 58000},
    "tt0301383": {"titre": "La Chute du faucon noir", "annee": 2001, "rating": "liked", "genres": ["War", "Drama", "Action"], "note": 7.7, "votes": 398000},
    "tt1655246": {"titre": "Démineur", "annee": 2008, "rating": "liked", "genres": ["War", "Drama", "Thriller"], "note": 7.5, "votes": 453000},
    "tt0076789": {"titre": "La 7ème compagnie", "annee": 1973, "rating": "liked", "genres": ["War", "Comedy"], "note": 6.9, "votes": 4000},
    "tt0120815": {"titre": "Il faut sauver le soldat Ryan", "annee": 1998, "rating": "liked", "genres": ["War", "Drama"], "note": 8.6, "votes": 1400000},
    "tt0062653": {"titre": "La Grande Vadrouille", "annee": 1966, "rating": "liked", "genres": ["War", "Comedy", "Adventure"], "note": 7.9, "votes": 18000},
    "tt0338564": {"titre": "Troie", "annee": 2004, "rating": "liked", "genres": ["War", "Adventure", "Drama"], "note": 7.3, "votes": 541000},
    "tt0408306": {"titre": "Apocalypto", "annee": 2006, "rating": "liked", "genres": ["Action", "Adventure", "Drama"], "note": 7.8, "votes": 319000},
    "tt0093058": {"titre": "Full Metal Jacket", "annee": 1987, "rating": "liked", "genres": ["War", "Drama"], "note": 8.3, "votes": 768000},
    
    # Films d'Animation (8)
    "tt0441773": {"titre": "Kung Fu Panda", "annee": 2008, "rating": "liked", "genres": ["Animation", "Action", "Adventure"], "note": 7.6, "votes": 490000},
    "tt0892769": {"titre": "Dragons", "annee": 2010, "rating": "liked", "genres": ["Animation", "Action", "Adventure"], "note": 8.1, "votes": 756000},
    "tt1431045": {"titre": "Dragons 2", "annee": 2014, "rating": "liked", "genres": ["Animation", "Action", "Adventure"], "note": 7.8, "votes": 343000},
    "tt2386490": {"titre": "Dragons 3", "annee": 2019, "rating": "liked", "genres": ["Animation", "Action", "Adventure"], "note": 7.4, "votes": 135000},
    "tt1049413": {"titre": "Kung Fu Panda 2", "annee": 2011, "rating": "liked", "genres": ["Animation", "Action", "Adventure"], "note": 7.3, "votes": 295000},
    "tt2267968": {"titre": "Kung Fu Panda 3", "annee": 2016, "rating": "liked", "genres": ["Animation", "Action", "Adventure"], "note": 7.1, "votes": 165000},
    "tt0401792": {"titre": "Là-haut", "annee": 2009, "rating": "liked", "genres": ["Animation", "Adventure", "Comedy"], "note": 8.3, "votes": 1100000},
    "tt1217209": {"titre": "Raiponce", "annee": 2010, "rating": "liked", "genres": ["Animation", "Adventure", "Comedy"], "note": 7.7, "votes": 456000},
    
    # Films Historiques/Biographiques (13)
    "tt0382932": {"titre": "Écrire pour exister", "annee": 2007, "rating": "liked", "genres": ["Biography", "Drama"], "note": 7.6, "votes": 92000},
    "tt2980516": {"titre": "L'incroyable histoire du temps", "annee": 2014, "rating": "liked", "genres": ["Biography", "Drama", "Romance"], "note": 7.7, "votes": 479000},
    "tt1205489": {"titre": "Le Discours d'un roi", "annee": 2010, "rating": "liked", "genres": ["Biography", "Drama", "History"], "note": 8.0, "votes": 706000},
    "tt0268978": {"titre": "Un homme d'exception", "annee": 2001, "rating": "liked", "genres": ["Biography", "Drama"], "note": 8.2, "votes": 975000},
    "tt1745960": {"titre": "Imitation Game", "annee": 2014, "rating": "liked", "genres": ["Biography", "Drama", "Thriller"], "note": 8.0, "votes": 788000},
    "tt0443272": {"titre": "Lincoln", "annee": 2012, "rating": "liked", "genres": ["Biography", "Drama", "History"], "note": 7.3, "votes": 279000},
    "tt0112573": {"titre": "Braveheart", "annee": 1995, "rating": "liked", "genres": ["Biography", "Drama", "History"], "note": 8.3, "votes": 1050000},
    "tt0172495": {"titre": "Gladiator", "annee": 2000, "rating": "liked", "genres": ["Action", "Adventure", "Drama"], "note": 8.5, "votes": 1600000},
    "tt0245429": {"titre": "USS Indianapolis", "annee": 2016, "rating": "liked", "genres": ["War", "Drama", "History"], "note": 5.3, "votes": 7000},
    "tt1028528": {"titre": "Walkyrie", "annee": 2008, "rating": "liked", "genres": ["Drama", "History", "Thriller"], "note": 7.1, "votes": 267000},
    "tt0367279": {"titre": "La Chute", "annee": 2004, "rating": "liked", "genres": ["Biography", "Drama", "History"], "note": 8.2, "votes": 367000},
}


def init_paul_profile_if_needed(user_manager, force=False):
    """
    Initialise automatiquement le profil de Paul s'il est vide
    
    Args:
        user_manager: Instance de UserManager
        force: Si True, réinitialise même si le profil existe
    
    Returns:
        bool: True si le profil a été initialisé
    """
    from datetime import datetime
    
    profile = user_manager.load_profile("paul")
    
    # Vérifier si le profil est vide
    if force or len(profile.get('films_vus', {})) == 0:
        films_vus = {}
        for tconst, film_data in PAUL_LIKED_FILMS.items():
            films_vus[tconst] = {
                "titre": film_data['titre'],
                "annee": film_data['annee'],
                "rating": film_data['rating'],
                "date_ajout": datetime.now().isoformat(),
                "genres": film_data['genres'],
                "note": film_data.get('note'),
                "votes": film_data.get('votes')
            }
        
        profile['films_vus'] = films_vus
        user_manager.save_profile("paul", profile)
        
        return True
    
    return False

