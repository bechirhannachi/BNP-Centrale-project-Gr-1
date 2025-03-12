import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
# Charger le fichier fusionné
df = pd.read_csv("fusion.csv")

# Remplir uniquement les colonnes numériques avec la médiane
df_numeric = df.select_dtypes(include=["number"])  # Sélectionne seulement les colonnes numériques
# On bouche les trous par la médiane car c'est plus robuste que la moyenne
df[df_numeric.columns] = df_numeric.fillna(df_numeric.median())





# Étape 1: One-Hot Encoding de la variable cible
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
category_encoded = encoder.fit_transform(df[['secteur_activite']])

# Ajouter les colonnes encodées dans le DataFrame
category_columns = encoder.get_feature_names_out(['secteur_activite'])
df_encoded = pd.DataFrame(category_encoded, columns=category_columns)

# Fusionner avec les features numériques
df_combined = pd.concat([df[['surface_utile', 'shon','estimation_ges','consommation_energie']], df_encoded], axis=1)

# Étape 2: Utilisation de KNNImputer
imputer = KNNImputer(n_neighbors=3)
df_imputed = pd.DataFrame(imputer.fit_transform(df_combined), columns=df_combined.columns)

# Étape 3: Reconvertir les colonnes One-Hot en catégorie
df_imputed['secteur_activite'] = encoder.inverse_transform(df_imputed[category_columns])

# Garder uniquement les colonnes originales
df_final = df[['surface_utile', 'shon','estimation_ges','consommation_energie']].copy()
df_final['secteur_activite'] = df_imputed['secteur_activite']




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
