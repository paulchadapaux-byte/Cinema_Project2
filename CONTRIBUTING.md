# 🤝 Guide de Contribution - Cinema Creuse

Guide pour contribuer au projet et ajouter de nouvelles fonctionnalités.

---

## 📋 Avant de Commencer

### Prérequis
- Python 3.9+
- Git
- Connaissance de Streamlit
- Connaissance de Pandas

### Installation
```bash
git clone [url-du-repo]
cd Cinema_Project2
pip install -r requirements.txt
```

---

## 🏗️ Architecture du Projet

### Fichiers Principaux

```
app/
├── main.py              # Application principale (UI + logique)
├── utils.py             # Fonctions utilitaires (recommandations, API, graphiques)
├── films_cache.py       # Cache films à l'affiche (généré auto)
└── user_profiles.json   # Profils utilisateurs (généré auto)
```

### Structure main.py

```python
# 1. Imports et Configuration
# 2. Fonctions de chargement (load_imdb_data, load_excel_data)
# 3. Sidebar (navigation + authentification)
# 4. Pages principales
#    - Page Profil Utilisateur (👤)
#    - Page Recommandations (🔍)
#    - Page Cinémas Creuse (🗺️)
#    - Espace B2B (📊)
```

### Structure utils.py

```python
# 1. Configuration (couleurs, API keys)
# 2. Gestion utilisateurs (UserProfileManager)
# 3. API TMDb (search, details, now_playing)
# 4. Système de recommandations (KNN, similarité)
# 5. Fonctions graphiques (Plotly, Matplotlib)
# 6. Carte Folium (cinémas)
```

---

## ✨ Ajouter une Nouvelle Fonctionnalité

### Exemple : Ajouter un Nouvel Onglet

#### 1. Dans main.py

```python
# Ajouter dans la sidebar
page = st.sidebar.radio(
    "Navigation",
    ["👤 Profil", "🔍 Recommandations", "🗺️ Cinémas", "📊 B2B", "🆕 Nouvelle Page"]
)

# Ajouter la logique
if page == "🆕 Nouvelle Page":
    st.title("🆕 Ma Nouvelle Page")
    
    # Charger les données si nécessaire
    df_movies = load_imdb_data()
    
    # Votre code ici
    st.write("Contenu de la page")
```

#### 2. Dans utils.py (si besoin de nouvelles fonctions)

```python
def ma_nouvelle_fonction(df, parametre):
    """
    Description de la fonction
    
    Args:
        df: DataFrame
        parametre: Description du paramètre
    
    Returns:
        Résultat
    """
    # Code ici
    return resultat
```

---

### Exemple : Ajouter un Nouveau Filtre

```python
# Dans main.py, dans la section des filtres

# Ajouter le filtre dans la sidebar ou dans la page
filtre_realisateur = st.selectbox(
    "Réalisateur",
    ["Tous"] + sorted(df_movies['realisateurs'].explode().unique())
)

# Appliquer le filtre
if filtre_realisateur != "Tous":
    df_filtered = df_movies[
        df_movies['realisateurs'].apply(lambda x: filtre_realisateur in x)
    ]
```

---

### Exemple : Ajouter un Nouveau Graphique B2B

```python
# Dans utils.py

def plot_nouveau_graphique(df):
    """Crée un nouveau graphique pour l'analyse B2B"""
    import plotly.express as px
    
    fig = px.bar(
        df,
        x='colonne_x',
        y='colonne_y',
        color='categorie',
        color_discrete_map=PALETTE_CREUSE
    )
    
    fig.update_layout(
        title="Titre du Graphique",
        xaxis_title="Axe X",
        yaxis_title="Axe Y"
    )
    
    return fig

# Dans main.py, dans la section B2B

fig = plot_nouveau_graphique(df_data)
st.plotly_chart(fig, use_container_width=True)
```

---

## 🔧 Bonnes Pratiques

### Code

1. **Docstrings** : Toujours documenter les fonctions
   ```python
   def ma_fonction(parametre):
       """
       Description courte
       
       Args:
           parametre (type): Description
       
       Returns:
           type: Description
       """
       pass
   ```

2. **Type Hints** (optionnel mais recommandé)
   ```python
   def ma_fonction(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
       pass
   ```

3. **Noms explicites**
   ```python
   # ❌ Mauvais
   def f(x, y):
       pass
   
   # ✅ Bon
   def calculate_similarity_score(movie1, movie2):
       pass
   ```

### Streamlit

1. **Cache** : Utiliser `@st.cache_data` pour fonctions lourdes
   ```python
   @st.cache_data
   def load_data():
       return pd.read_parquet('data.parquet')
   ```

2. **Session State** : Pour garder l'état entre reruns
   ```python
   if 'compteur' not in st.session_state:
       st.session_state.compteur = 0
   
   if st.button("Incrémenter"):
       st.session_state.compteur += 1
   ```

3. **Spinners** : Pour indiquer chargement
   ```python
   with st.spinner("Chargement..."):
       resultat = fonction_longue()
   ```

### Git

1. **Commits clairs**
   ```bash
   # ✅ Bon
   git commit -m "feat: Add new filter for directors"
   git commit -m "fix: Correct B2B column names"
   git commit -m "docs: Update README with new features"
   
   # ❌ Mauvais
   git commit -m "update"
   git commit -m "fix bug"
   ```

2. **Branches** : Une branche par fonctionnalité
   ```bash
   git checkout -b feature/nouvelle-fonctionnalite
   # Développer
   git add .
   git commit -m "feat: Description"
   git push origin feature/nouvelle-fonctionnalite
   ```

---

## 🧪 Tester ses Modifications

### Test Local

```bash
# Lancer l'app
cd app
streamlit run main.py

# Tester dans le navigateur
# - Créer un compte
# - Tester les filtres
# - Vérifier les graphiques
# - Tester sur différents navigateurs
```

### Checklist

- [ ] Le code fonctionne sans erreur
- [ ] Les filtres fonctionnent correctement
- [ ] Les graphiques s'affichent
- [ ] La navigation est fluide
- [ ] Pas de ralentissement
- [ ] Le cache fonctionne
- [ ] Les profils sont sauvegardés

---

## 📊 Ajouter de Nouvelles Données

### Nouveau Fichier Excel

```python
# Dans utils.py ou main.py

@st.cache_data
def load_new_data():
    """Charge les nouvelles données"""
    excel_path = DATA_DIR / 'nouveau_fichier.xlsx'
    df = pd.read_excel(excel_path, sheet_name='Feuille1')
    return df
```

### Nouvelle Colonne dans IMDb

```python
# Si tu ajoutes une colonne dans le notebook de préparation
# Elle sera automatiquement disponible dans df_movies

# Utilisation
if 'nouvelle_colonne' in df_movies.columns:
    valeur = df_movies['nouvelle_colonne'].iloc[0]
```

---

## 🎨 Personnaliser le Style

### Couleurs

```python
# Dans utils.py
PALETTE_CREUSE = {
    'principal': '#2F5233',
    'secondaire': '#5D8A66',
    'accent': '#D4AF37',
    'nouveau': '#123456'  # Ajouter ici
}
```

### CSS Custom

```python
# Dans main.py
st.markdown("""
    <style>
    .ma-classe {
        color: #2F5233;
        font-size: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Utilisation
st.markdown('<div class="ma-classe">Mon texte</div>', unsafe_allow_html=True)
```

---

## 🐛 Déboguer

### Messages de Debug

```python
# Pendant le développement
st.write("DEBUG:", variable)
st.write(df.head())
st.write(df.columns)
```

### Logs dans le Terminal

```python
print("DEBUG:", variable)  # S'affiche dans le terminal
```

### Try/Except

```python
try:
    resultat = fonction_risquee()
except Exception as e:
    st.error(f"Erreur : {e}")
    print(f"Erreur détaillée : {e}")  # Dans le terminal
```

---

## 📝 Documenter

### README

Mettre à jour `README.md` si :
- Nouvelle fonctionnalité majeure
- Nouvelle dépendance
- Nouveau fichier requis

### Changelog

Mettre à jour `CHANGELOG.md` :
```markdown
## [Version X.Y] - 2026-XX-XX

### ✨ Nouvelles Fonctionnalités
- Description

### 🔧 Corrections
- Description
```

### Docstrings

Documenter les nouvelles fonctions dans le code

---

## 🤝 Workflow de Contribution

### 1. Créer une Branche

```bash
git checkout -b feature/ma-nouvelle-fonctionnalite
```

### 2. Développer

```bash
# Modifier les fichiers
# Tester localement
```

### 3. Commiter

```bash
git add .
git commit -m "feat: Add new feature"
```

### 4. Pousser

```bash
git push origin feature/ma-nouvelle-fonctionnalite
```

### 5. Créer une Pull Request

Sur GitHub/GitLab :
- Décrire les changements
- Ajouter des captures d'écran si UI
- Demander une review

### 6. Merger

Après validation :
- Merger dans main
- Supprimer la branche

---

## 📞 Support

Si tu as des questions :
1. Vérifier ce guide
2. Consulter la documentation Streamlit
3. Demander à l'équipe
4. Ouvrir une issue sur le repo

---

## ✅ Checklist avant de Commiter

- [ ] Code testé localement
- [ ] Pas d'erreurs dans la console
- [ ] Docstrings ajoutées
- [ ] README mis à jour si nécessaire
- [ ] CHANGELOG mis à jour
- [ ] Commit message clair
- [ ] Pas de données sensibles (mots de passe, clés API personnelles)

---

**Merci de contribuer au projet ! 🎬🚀**
