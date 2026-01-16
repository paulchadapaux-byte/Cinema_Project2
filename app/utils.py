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
                st.success("✅ Connexion réussie !")
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
