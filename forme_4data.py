import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
# Charger le fichier fusionné
df = pd.read_csv("fusion.csv")

# Remplir uniquement les colonnes numériques avec la médiane
df_numeric = df.select_dtypes(include=["number"])  # Sélectionne seulement les colonnes numériques
# On bouche les trous par la médiane car c'est plus robuste que la moyenne
df[df_numeric.columns] = df_numeric.fillna(df_numeric.median())


# Séparer les données avec et sans valeurs manquantes
# df_train = df.dropna(subset=['secteur_activite'])  # Lignes sans valeurs manquantes
# df_missing = df[df['secteur_activite'].isna()]    # Lignes à compléter

# # Encodage des catégories
# secteur_activite_map = {val: idx for idx, val in enumerate(df_train['secteur_activite'].unique())}
# df_train['secteur_activite_encoded'] = df_train['secteur_activite'].map(secteur_activite_map)

# # Entraînement du modèle KNN
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(df_train[['surface_utile', 'shon','estimation_ges','consommation_energie']], df_train['secteur_activite_encoded'])

# # Prédiction des valeurs manquantes
# df_missing['secteur_activite_encoded'] = knn.predict(df_missing[['surface_utile', 'shon','estimation_ges','consommation_energie']])

# # Décodage des catégories prédites
# reverse_secteur_activite_map = {v: k for k, v in secteur_activite_map.items()}
# df_missing['secteur_activite'] = df_missing['secteur_activite_encoded'].map(reverse_secteur_activite_map)

# # Réintégration des données complétées
# df_filled = pd.concat([df_train.drop(columns=['secteur_activite_encoded']), df_missing.drop(columns=['secteur_activite_encoded'])])


# Dictionnaire des valeurs possibles et des noms de fichiers associés
dict = {
    "Autres cas (par exemple: théâtres, salles de sport, restauration, commerces individuels, etc)": "data1.csv",
    "Bâtiment à occupation continue (par exemple: hopitaux, hôtels, internats, maisons de retraite, etc)": "data2.csv",
    "Centre commercial": "data3.csv",
    "Bâtiment à usage principale de bureau, d'administration ou d'enseignement": "data4.csv"
}

# Boucle pour filtrer et enregistrer chaque catégorie
for valeur, fichier in dict.items():
    df_filtre = df[df["secteur_activite"] == valeur]
    df_filtre.to_csv(fichier, index=False)
    print(f"{len(df_filtre)} lignes enregistrées dans {fichier}")
