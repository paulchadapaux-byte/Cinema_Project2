# 🎬 Cinema Creuse - Application de Recommandations de Films

Application Streamlit complète pour la recommandation de films et l'analyse du marché cinématographique dans la Creuse.

---

## 📋 Fonctionnalités

### 👤 Profil Utilisateur
- ✅ Création de compte et connexion
- ✅ Gestion des films aimés/pas aimés (👍/👎)
- ✅ **Recommandations personnalisées** basées sur vos goûts
- ✅ Films actuellement à l'affiche en France (API TMDb)
- ✅ Films à venir prochainement
- ✅ Historique de vos préférences

### 🔍 Recherche Manuelle
- ✅ Recherche par **titre de film**
- ✅ Recherche par **acteur**
- ✅ Recherche par **réalisateur**
- ✅ Films similaires (bouton "Voir les recommandations")
- ✅ Affichage en grille avec posters TMDb
- ✅ Synopsis complet, casting, réalisateur

### 🗺️ Cinémas de la Creuse
- ✅ Carte interactive avec 7 cinémas
- ✅ Localisation automatique de l'utilisateur
- ✅ Calcul des distances
- ✅ Films à l'affiche par cinéma
- ✅ Informations complètes (adresse, téléphone, horaires)

### 📊 Espace B2B
- ✅ Authentification sécurisée
- ✅ Analyses démographiques (population, âge, diplômes)
- ✅ Analyse de marché (fréquentation, genres préférés)
- ✅ Analyse concurrentielle (streaming vs cinéma)
- ✅ Matrice SWOT
- ✅ Export de données CSV

---

## 📂 Structure du Projet

```
Cinema_Project2/
├── app/
│   ├── main.py                    ← Application principale
│   ├── utils.py                   ← Fonctions utilitaires
│   ├── films_cache.py             ← Cache des films à l'affiche
│   └── user_profiles.json         ← Profils utilisateurs (généré automatiquement)
├── data/
│   ├── PARQUETS/
│   │   └── imdb_complet_avec_cast.parquet  ← Dataset IMDb (51K films France)
│   └── Cinemas_existants_creuse.xlsx       ← Données Excel
├── notebooks/
│   └── 00_data_preparation.ipynb  ← Préparation des données
├── requirements.txt               ← Dépendances Python
├── .gitignore                     ← Fichiers à ignorer
└── README.md                      ← Ce fichier
```

---

## ⚙️ Installation

### 1. Cloner le projet

```bash
git clone [url-du-repo]
cd Cinema_Project2
```

### 2. Créer un environnement virtuel (recommandé)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Générer le dataset IMDb

Si le fichier `imdb_complet_avec_cast.parquet` n'existe pas :

```bash
# Ouvrir Jupyter Notebook
jupyter notebook

# Ouvrir et exécuter notebooks/00_data_preparation.ipynb
# Durée : ~15-20 minutes
```

**Résultat attendu :**
```
✅ 51,582 films distribués en France conservés
💾 Sauvegardé : data/PARQUETS/imdb_complet_avec_cast.parquet
```

---

## 🚀 Lancement

```bash
cd app
streamlit run main.py
```

L'application s'ouvrira automatiquement à : **http://localhost:8501**

---

## 🌐 Déploiement sur Streamlit Cloud

### Déploiement en 3 Étapes

**1. Pousser sur GitHub**

```bash
git add .
git commit -m "Ready for deployment"
git push origin main
```

**2. Créer l'App sur Streamlit Cloud**

- Aller sur : https://share.streamlit.io/
- Se connecter avec GitHub
- Cliquer "New app"
- Configurer :
  - Repository : `TON-USERNAME/Cinema_Project2`
  - Branch : `main`
  - Main file : `app/main.py`
- Cliquer "Deploy!"

**3. Attendre 5-10 minutes**

Ton app sera accessible à :
```
https://TON-USERNAME-cinema-project2-app-main-XXXXXX.streamlit.app
```

### Vérification Avant Déploiement

```bash
# Lancer le script de vérification
python check_deployment.py
```

**Voir `DEPLOIEMENT_STREAMLIT.md` pour le guide complet.**

---

## 🔑 Identifiants B2B

Pour accéder à l'Espace B2B :

| Utilisateur | Mot de passe |
|-------------|--------------|
| paul        | WCS26        |
| hamidou     | WCS26        |
| lynda       | WCS26        |

---

## 🎯 Guide d'Utilisation

### Scénario 1 : Créer un Profil et Obtenir des Recommandations

```
1. Aller dans "👤 Profil utilisateur"
2. Créer un compte (ex: username: "john", password: "1234")
3. Marquer des films comme aimés (👍) ou pas aimés (👎)
4. Cliquer sur "✨ Afficher mes recommandations"
5. Voir les films personnalisés avec score de correspondance
6. Cliquer sur "📄 Voir le synopsis" pour plus de détails
```

### Scénario 2 : Films à l'Affiche

```
1. Aller dans "👤 Profil utilisateur"
2. Tab "🎬 Déjà en salles"
3. Voir les films actuellement au cinéma en France
4. Filtrer par genre ou note
5. Trier par popularité, note, titre
6. Cliquer sur "📄 Voir les détails" pour synopsis complet
```

### Scénario 3 : Recherche Manuelle

```
1. Aller dans "🔍 Recommandations"
2. Choisir le type de recherche :
   - "Film" → Chercher par titre
   - "Acteur" → Chercher par nom d'acteur
   - "Réalisateur" → Chercher par nom de réalisateur
3. Taper le nom (ex: "Brad Pitt")
4. Cliquer "Rechercher"
5. Cliquer "🎬 Voir les recommandations" sur un film
6. Voir le carousel de 6 films similaires
7. Cliquer "📄 Détails" pour voir le synopsis
```

### Scénario 4 : Trouver un Cinéma

```
1. Aller dans "🗺️ Cinémas Creuse"
2. Autoriser la géolocalisation (ou cliquer manuellement sur la carte)
3. Voir les 7 cinémas de la Creuse
4. Cliquer sur un marqueur pour voir les détails
5. Voir les films à l'affiche dans ce cinéma
6. Calculer l'itinéraire
```

### Scénario 5 : Analyse B2B

```
1. Aller dans "📊 Espace B2B"
2. Se connecter (ex: paul / WCS26)
3. Naviguer entre les 5 onglets :
   - Analyse de marché (démographie, post-COVID)
   - Analyse concurrentielle (prix, programmation)
   - Analyse interne (CSP, fréquentation)
   - SWOT (forces, faiblesses, opportunités, menaces)
   - Export (télécharger les données CSV)
4. Utiliser les boutons "Précédent/Suivant" pour naviguer
```

---

## 📊 Données

### Dataset IMDb (51,582 films)

**Source :** IMDb Datasets (https://datasets.imdbws.com/)

**Filtres appliqués :**
- Films distribués en France uniquement
- Années > 1990
- Avec notes IMDb
- Avec titres français
- Avec casting complet

**Colonnes principales :**
```
tconst, primaryTitle, originalTitle, frenchTitle, startYear, 
runtimeMinutes, genres, averageRating, numVotes, 
acteurs, realisateurs, isAdult
```

### API TMDb

**Source :** The Movie Database API (https://www.themoviedb.org/)

**Clé API :** `a8617cdd3b93f8a353f24a1843ccaafb`

**Fonctionnalités :**
- Films à l'affiche en France (now_playing)
- Films à venir (upcoming)
- Détails de films (posters, synopsis, casting)
- Recherche de films par titre/année
- Cache 24h pour optimiser les performances

### Données Excel

**Fichier :** `Cinemas_existants_creuse.xlsx`

**Feuilles :**
- Population_creuse : Répartition par âge/sexe
- Enfants_creuse : Types de familles
- Diplome_creuse : Niveau d'éducation
- Cine_Age_Global : Fréquentation par âge
- movies_type_shares : Préférences de genres
- prix_mensuel : Prix cinéma
- prix_streaming : Prix plateformes
- Confiseries : Évolution CA confiseries

---

## 🔧 Configuration

### Chemins de Fichiers

```python
# Dans main.py et utils.py
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Dataset IMDb
imdb_path = DATA_DIR / 'PARQUETS' / 'imdb_complet_avec_cast.parquet'

# Données Excel
excel_path = DATA_DIR / 'Cinemas_existants_creuse.xlsx'
```

### API TMDb

```python
# Dans utils.py
TMDB_API_KEY = "a8617cdd3b93f8a353f24a1843ccaafb"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
```

### Profils Utilisateurs

```python
# Fichier généré automatiquement
user_profiles_path = PROJECT_ROOT / "app" / "user_profiles.json"
```

---

## 🎨 Personnalisation

### Couleurs

```python
# Dans utils.py
PALETTE_CREUSE = {
    'principal': '#2F5233',    # Vert foncé
    'secondaire': '#5D8A66',   # Vert moyen
    'accent': '#D4AF37',       # Or
    'bleu': '#3498DB',         # Bleu
    'rouge': '#E74C3C'         # Rouge
}
```

### Nombre de Recommandations

```python
# Dans main.py
# Recommandations manuelles (recherche)
reco_df, method = get_recommendations(df_movies, selected_idx, n=6)

# Recommandations personnalisées
recommended_films = get_personalized_recommendations(
    df_movies, liked_films, disliked_films, top_n=20
)
```

---

## 🐛 Résolution de Problèmes

### Erreur "Fichier non trouvé"

```bash
# Vérifier que le fichier parquet existe
ls data/PARQUETS/imdb_complet_avec_cast.parquet

# Si absent, relancer le notebook
jupyter notebook
# Ouvrir 00_data_preparation.ipynb
# Kernel > Restart & Run All
```

### Erreur "118,277 films au lieu de 51,582"

```bash
# Effacer le cache Streamlit
streamlit cache clear

# Relancer l'app
streamlit run main.py
```

### Erreur API TMDb

```python
# Vérifier la clé API dans utils.py
TMDB_API_KEY = "a8617cdd3b93f8a353f24a1843ccaafb"

# Vérifier la connexion Internet
# Les fallbacks automatiques affichent des placeholders si échec
```

### Synopsis Manquants

```bash
# S'assurer d'utiliser la dernière version de main.py
# Le synopsis est dans un expander "📄 Voir le synopsis"
# ou "📄 Voir les détails" ou "📄 Plus d'infos"
```

---

## 📈 Performance

| Opération | Durée |
|-----------|-------|
| Chargement initial | 2-5 secondes |
| Recherche de films | <1 seconde |
| Calcul recommandations | 1-2 secondes |
| Enrichissement TMDb (6 films) | 2-3 secondes |
| Chargement films à l'affiche | 3-5 secondes (1ère fois) |
| Cache TMDb | 24 heures |

---

## 🛠️ Technologies Utilisées

| Technologie | Version | Usage |
|-------------|---------|-------|
| Python | 3.9+ | Langage principal |
| Streamlit | 1.29.0+ | Framework web |
| Pandas | 2.1+ | Manipulation de données |
| Scikit-learn | 1.3+ | Machine Learning (KNN) |
| Folium | 0.15+ | Cartes interactives |
| Plotly | 5.18+ | Graphiques interactifs |
| Requests | 2.31+ | Appels API |
| PyArrow | 14.0+ | Lecture Parquet |
| OpenPyXL | 3.1+ | Lecture Excel |

---

## 📝 TODO / Améliorations Futures

- [ ] Ajouter plus de cinémas de la région
- [ ] Intégration avec l'API Allociné pour horaires réels
- [ ] Système de notation des films
- [ ] Export PDF des recommandations
- [ ] Statistiques avancées du profil utilisateur
- [ ] Partage de profils entre utilisateurs
- [ ] Mode sombre / Mode clair
- [ ] Application mobile

---

## 👥 Auteurs

**Équipe Wild Code School 2026 :**
- Paul (Data Analyst)
- Hamidou
- Lynda

---

## 📄 Licence

Ce projet est développé dans le cadre de la formation Wild Code School.

---

## 🆘 Support

Pour toute question ou problème :

1. Vérifier le README
2. Consulter les messages d'erreur Streamlit
3. Vérifier les chemins de fichiers
4. S'assurer que toutes les dépendances sont installées
5. Effacer le cache : `streamlit cache clear`

---

**Bon cinéma ! 🎬🍿**
