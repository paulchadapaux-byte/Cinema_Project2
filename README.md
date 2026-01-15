# Ìæ¨ Projet IMDb - Syst√®me de Recommandation

**Wild Code School - Projet 2 - Data Analyst**

## Ì≥ä Description

Syst√®me de recommandation de films bas√© sur les donn√©es IMDb, int√©grant :
- Analyse exploratoire des donn√©es
- Feature engineering (genres, acteurs, r√©alisateurs)
- Mod√®le de recommandation content-based
- Application Streamlit interactive

## Ì∑ÇÔ∏è Structure du projet
```
PROJET_2_IMDB/
‚îú‚îÄ‚îÄ data/                    # Donn√©es
‚îú‚îÄ‚îÄ notebooks/               # Analyses Jupyter
‚îú‚îÄ‚îÄ src/                     # Code source
‚îú‚îÄ‚îÄ app/                     # Application Streamlit
‚îî‚îÄ‚îÄ docs/                    # Documentation
```

## Ì∫Ä Installation

### 1. Cloner le repository
```bash
git clone https://github.com/ton-username/projet-imdb.git
cd projet-imdb
```

### 2. Cr√©er un environnement virtuel
```bash
python -m venv env
source env/bin/activate  # Linux/Mac
env\Scripts\activate     # Windows
```

### 3. Installer les d√©pendances
```bash
pip install -r requirements.txt
```

### 4. T√©l√©charger les donn√©es
Les donn√©es sont trop volumineuses pour GitHub.

**Option A : T√©l√©charger depuis Google Drive**
- [Lien vers les donn√©es](https://drive.google.com/...)
- Placer dans `data/processed/`

**Option B : R√©g√©n√©rer les donn√©es**
```bash
jupyter notebook notebooks/00_data_preparation.ipynb
```

## Ì≥ì Notebooks

1. `00_data_preparation.ipynb` - Nettoyage et pr√©paration des donn√©es IMDb
2. `01_EDA.ipynb` - Analyse exploratoire
3. `02_feature_engineering.ipynb` - Cr√©ation des features
4. `03_recommandation.ipynb` - D√©veloppement du syst√®me de reco

## ÌæØ Lancer l'application
```bash
streamlit run app/streamlit_app.py
```

## Ì≥¶ Donn√©es

- **Source** : [IMDb Datasets](https://datasets.imdbws.com/)
- **P√©riode** : Films de 1980 √† 2024
- **Taille** : ~500k films
- **Features** : Genres, acteurs, r√©alisateurs, notes, votes

## Ìª†Ô∏è Technologies

- Python 3.10+
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Matplotlib, Seaborn

## Ì±§ Auteur

Paul - Data Analyst @ Wild Code School

## Ì≥ù License

Projet √©ducatif - Wild Code School# Cinema_Project2
Projet 2 Wild Code School Cinema Creuse
