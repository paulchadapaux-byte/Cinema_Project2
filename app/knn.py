from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline


#Séparer les features
meta= ['tconst', 'titleType', 'primaryTitle', 'originalTitle', 'isAdult', 'genres', 'frenchTitle']
X= df_movie.drop(meta, axis= 1)

#Standardiser en séparant les valeurs binaires / numériques
binary_columns= X.loc[:, X.nunique()== 2].columns
numerical_columns= X.drop(binary_columns, axis= 1).columns

#scaler sur les colonnes numériques (runtimeMinutes, startYear, averageRating, numVotes)
#passthrough sur les colonnes binaires (genres)
preprocessor = ColumnTransformer(
    transformers=[
        ('binary', 'passthrough', binary_columns),
        ('numerical', StandardScaler(), numerical_columns)
    ]
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('KNN', KNearestNeighbors(n_neighbors=21, metric= 'cosine'))
])

pipeline.fit_transform(X)

y_pred= X['tconst'][1:21]