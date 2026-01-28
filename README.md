# 🎬 Cinema Creuse - Système de recommandation de films

Application Streamlit de recommandation de films utilisant l'algorithme KNN (K-Nearest Neighbors) avec enrichissement TMDb.

## 🎯 Fonctionnalités

- **Recommandations KNN** : Films similaires basés sur 177 features (genres, réalisateurs, acteurs)
- **Films à l'affiche** : Actualisation quotidienne via API TMDb
- **Recherche avancée** : Par film, acteur, réalisateur, ou année
- **Profils utilisateurs** : Sauvegarde des films aimés/pas aimés
- **Visualisations** : Graphiques interactifs et statistiques

## 🚀 Installation

### 1. Cloner le repository

```bash
git clone https://github.com/TON_USERNAME/Cinema_Creuse.git
cd Cinema_Creuse
```

### 2. Créer un environnement virtuel (recommandé)

```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer la clé API TMDb

**Obtenir une clé API (gratuit) :**
1. Créer un compte sur [The Movie Database](https://www.themoviedb.org)
2. Aller dans Settings → API
3. Demander une clé API (3000 requêtes/jour gratuites)

**Configurer l'application :**
```bash
# Copier le template
cp .streamlit/secrets.toml.example .streamlit/secrets.toml

# Éditer et ajouter votre clé
nano .streamlit/secrets.toml
```

Remplacer `VOTRE_CLE_ICI` par votre vraie clé API.

### 5. Préparer les données

L'application nécessite un fichier parquet IMDb avec les films :

```bash
# Créer le dossier DATA
mkdir -p DATA/PARQUETS

# Placer votre fichier
# DATA/PARQUETS/imdb_complet_avec_cast.parquet
```

### 6. Lancer l'application

```bash
streamlit run main.py
```

L'application s'ouvre automatiquement dans votre navigateur à `http://localhost:8501`

## 📊 Structure du projet

```
Cinema_Creuse/
├── main.py                      # Application principale Streamlit
├── utils.py                     # Fonctions utilitaires
├── films_cache.py              # Cache films à l'affiche
├── knn.py                      # Modèle KNN (optionnel)
├── user_manager.py             # Gestion profils utilisateurs
├── requirements.txt            # Dépendances Python
├── .gitignore                  # Fichiers ignorés par Git
├── .streamlit/
│   ├── secrets.toml.example    # Template configuration
│   └── secrets.toml            # Votre configuration (non committé)
├── DATA/
│   └── PARQUETS/
│       └── imdb_complet_avec_cast.parquet
└── data/
    └── user_profiles/          # Profils utilisateurs sauvegardés
```

## 🔐 Sécurité

⚠️ **Important** : Ne JAMAIS committer votre clé API TMDb !

Le fichier `.streamlit/secrets.toml` contient votre clé et est déjà dans `.gitignore`.

## 🎓 Algorithme KNN

L'application utilise un système de recommandation basé sur KNN avec :
- **177 features** : genres (25), réalisateurs (50), acteurs (100), année + durée (2)
- **Distance cosine** : Mesure de similarité entre films
- **Pipeline sklearn** : Preprocessing avec ColumnTransformer + MultiLabelBinarizer

**3 modes de recommandation** :
1. **Par film** : Films similaires à un film donné
2. **Par acteur** : Filmographie similaire d'un acteur
3. **Favoris** : Recommandations personnalisées (vecteur moyen)

## 📈 Technologies utilisées

- **Streamlit** : Interface web
- **Scikit-learn** : Algorithme KNN
- **Pandas/Numpy** : Manipulation de données
- **TMDb API** : Enrichissement films
- **Matplotlib/Seaborn** : Visualisations

## 👥 Auteur

Projet réalisé dans le cadre de la formation Data Analyst à Wild Code School.

## 📝 Licence

MIT License

## 🆘 Support

Pour toute question ou problème :
1. Vérifier que `.streamlit/secrets.toml` existe et contient votre clé
2. Vérifier que le fichier parquet est présent dans `DATA/PARQUETS/`
3. Vider le cache Streamlit : touche `C` dans l'application

---

**Date de dernière mise à jour** : Janvier 2026
