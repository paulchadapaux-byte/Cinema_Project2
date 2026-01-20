"""
Gestionnaire de profils utilisateurs
Permet de sauvegarder et récupérer les films vus/aimés par chaque utilisateur
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime


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
    
    def save_profile(self, username, profile):
        """
        Sauvegarde le profil d'un utilisateur
        
        Args:
            username: Nom d'utilisateur
            profile: Dictionnaire du profil à sauvegarder
        """
        profile_path = self._get_profile_path(username)
        
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)
    
    def add_film(self, username, film_data, rating):
        """
        Ajoute un film au profil de l'utilisateur
        
        Args:
            username: Nom d'utilisateur
            film_data: Dictionnaire avec les infos du film (doit contenir 'tconst' ou 'imdb_id')
            rating: 'liked' ou 'disliked'
        
        Returns:
            bool: True si succès
        """
        profile = self.load_profile(username)
        
        # Récupérer l'ID du film (peut être tconst ou imdb_id selon la source)
        film_id = film_data.get('tconst') or film_data.get('imdb_id') or film_data.get('id')
        
        if not film_id:
            return False
        
        # Ajouter le film
        profile['films_vus'][str(film_id)] = {
            "titre": film_data.get('titre', 'Titre inconnu'),
            "annee": film_data.get('startYear') or film_data.get('annee'),
            "rating": rating,
            "date_ajout": datetime.now().isoformat(),
            "genres": film_data.get('genres', []),
            "note": film_data.get('note'),
            "votes": film_data.get('votes')
        }
        
        self.save_profile(username, profile)
        return True
    
    def update_film_rating(self, username, film_id, new_rating):
        """
        Modifie la note d'un film (passer de liked à disliked ou inversement)
        
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
            film_id: ID du film à supprimer
        
        Returns:
            bool: True si succès
        """
        profile = self.load_profile(username)
        
        if str(film_id) in profile['films_vus']:
            del profile['films_vus'][str(film_id)]
            self.save_profile(username, profile)
            return True
        
        return False
    
    def get_films_by_rating(self, username, rating=None):
        """
        Récupère les films vus par l'utilisateur, filtrés par note
        
        Args:
            username: Nom d'utilisateur
            rating: 'liked', 'disliked' ou None (tous)
        
        Returns:
            list: Liste de tuples (film_id, film_data)
        """
        profile = self.load_profile(username)
        films = []
        
        for film_id, film_data in profile['films_vus'].items():
            if rating is None or film_data.get('rating') == rating:
                films.append((film_id, film_data))
        
        # Trier par date d'ajout (plus récent d'abord)
        films.sort(key=lambda x: x[1].get('date_ajout', ''), reverse=True)
        
        return films
    
    def get_liked_films(self, username):
        """Récupère les films aimés"""
        return self.get_films_by_rating(username, 'liked')
    
    def get_disliked_films(self, username):
        """Récupère les films pas aimés"""
        return self.get_films_by_rating(username, 'disliked')
    
    def get_all_films(self, username):
        """Récupère tous les films vus"""
        return self.get_films_by_rating(username, None)
    
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
                for genre in film['genres']:
                    if isinstance(genre, str):
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
        """
        Vérifie si un film a déjà été noté par l'utilisateur
        
        Args:
            username: Nom d'utilisateur
            film_id: ID du film
        
        Returns:
            str: 'liked', 'disliked' ou None
        """
        profile = self.load_profile(username)
        
        if str(film_id) in profile['films_vus']:
            return profile['films_vus'][str(film_id)].get('rating')
        
        return None
    
    def get_preferred_genres(self, username, top_n=5):
        """
        Retourne les genres préférés de l'utilisateur (basés sur les films aimés)
        
        Args:
            username: Nom d'utilisateur
            top_n: Nombre de genres à retourner
        
        Returns:
            list: Liste des genres préférés
        """
        liked_films = self.get_liked_films(username)
        
        genres_count = {}
        for _, film_data in liked_films:
            if film_data.get('genres'):
                for genre in film_data['genres']:
                    if isinstance(genre, str):
                        genres_count[genre] = genres_count.get(genre, 0) + 1
        
        # Trier par popularité
        sorted_genres = sorted(genres_count.items(), key=lambda x: x[1], reverse=True)
        
        return [genre for genre, count in sorted_genres[:top_n]]
