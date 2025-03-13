# %% Importer les modules
import pandas as pd
from pyproj import Transformer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np

# %% Mapping sur la base 1
df1=pd.read_csv('dpe1.csv')
filter1=(df1['date_visite_diagnostiqueur']>'1677-09-21') & (df1['date_visite_diagnostiqueur']<'2026')
filter2=(df1['date_etablissement_dpe']>'1677-09-21') & (df1['date_etablissement_dpe']<'2026')
filter3=(df1['date_arrete_tarifs_energies']>'1677-09-21') & (df1['date_arrete_tarifs_energies']<'2026')
df1=df1[filter1&filter2&filter3]
df1['date_visite_diagnostiqueur']=pd.to_datetime(df1['date_visite_diagnostiqueur'])
df1['date_etablissement_dpe']=pd.to_datetime(df1['date_etablissement_dpe'])
df1['date_arrete_tarifs_energies']=pd.to_datetime(df1['date_arrete_tarifs_energies'])
def modify(ch):
    # Si ch est une chaîne de caractères, on effectue le split
    if isinstance(ch, str):
        return ch.split("\xa0")[0]+' '+ch.split("\xa0")[1]
    else:
        # Si ce n'est pas une chaîne, on retourne la valeur telle quelle
        return ch
df1['tr013_type_erp_type']=df1['tr013_type_erp_type'].apply(modify)
df_names=pd.read_excel('mapping_secteur_activite.xlsx')
# Création du dictionnaire de correspondance
mapping = df_names.set_index('secteur_activite')['asset_type_cre'].to_dict()

# Remplacement des valeurs avec gestion des valeurs absentes
df1['secteur_activite'] = df1['tr013_type_erp_type'].map(mapping).fillna(df1['secteur_activite'])
df1['secteur_activite'] = df1['secteur_activite'].map(mapping).fillna(df1['secteur_activite'])
#Dictionnaire pour supprimer les valeurs non changées:
mapping2 = {key:key for key in df_names['asset_type_cre'].unique()}
df1['secteur_activite'] = df1['secteur_activite'].map(mapping2)
df1.dropna(subset=['secteur_activite'], inplace=True)

#%%  Mapping sur la base 2
df2 = pd.read_csv('dpe2.csv')
df2['Secteur_activité'] = df2['Secteur_activité'].map(mapping)
df2.dropna(subset=['Secteur_activité'], inplace=True)

#%%  Transformer les longitudes, latitudes en X,Y sur la base 1
transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
df1[['X_L93', 'Y_L93']] = df1.apply(lambda row: transformer.transform(row['longitude'], row['latitude']), axis=1, result_type='expand')

#%%  Uniformiser le nom des variables sur les deux bases

correspondances = {
    "surface_utile": "Surface_utile",
    "shon": "Surface_(SHON)",
    "date_etablissement_dpe" : "Date_établissement_DPE",
    "annee_construction": "Année_construction",
    "secteur_activite": "Secteur_activité",
    #"code_postal": "Code_postal_(brut)",  # il faut voir si c pas mieux de mettre code postal ban
    "classe_estimation_ges" : "Etiquette_GES",
    "estimation_ges" : "Emission_GES_kgCO2/m²/an",
    "consommation_energie" : "Conso_kWhep/m²/an",
    "X_L93" : "Coordonnée_cartographique_X_(BAN)",
    "Y_L93" : "Coordonnée_cartographique_Y_(BAN)"
}
df2 = df2.rename(columns={v: k for k, v in correspondances.items()})
# Garder uniquement les colonnes du dictionnaire
colonnes_finales = list(correspondances.keys())
df1 = df1[colonnes_finales]
df2 = df2[colonnes_finales]

#%%  Gérer les valeurs numériques
colonnes_numeriques = ["surface_utile", "shon", "estimation_ges", "consommation_energie", "X_L93", "Y_L93"]
for col in colonnes_numeriques:
    df1[col] = pd.to_numeric(df1[col], errors="coerce")
    df2[col] = pd.to_numeric(df2[col], errors="coerce")

#%%  Fusionner les bases et enlever les valeurs aberrantes
# Concaténer les DataFrames verticalement
df_merged = pd.concat([df1, df2], ignore_index=True)

# Filtrer les valeurs aberrantes
df_merged = df_merged[df_merged["estimation_ges"] >= 0]
df_merged = df_merged[df_merged["surface_utile"] >= 0]
df_merged = df_merged[df_merged["shon"] >= 0]
df_merged = df_merged[df_merged["consommation_energie"] >= 0]
colonnes_finales = list(correspondances.keys())
df= df_merged[colonnes_finales]


#%%  Imputation des valeurs manquantes

# Remplir uniquement les colonnes numériques avec la médiane
df_numeric = df.select_dtypes(include=["number"])  # Sélectionne seulement les colonnes numériques
cols_to_replace = ['shon','consommation_energie','estimation_ges']  # Liste des colonnes où traiter les 0 comme NaN
df[cols_to_replace] = df[cols_to_replace].replace(0, np.nan) #On considère les 0 comme des valeurs manquantes
# On bouche les trous par la médiane car c'est plus robuste que la moyenne
df[df_numeric.columns] = df_numeric.fillna(df_numeric.median())


# Étape 1: One-Hot Encoding de la variable cible, centrer et normaliser les variables numériques 
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
category_encoded = encoder.fit_transform(df[['secteur_activite']])

# Ajouter les colonnes encodées dans le DataFrame
category_columns = encoder.get_feature_names_out(['secteur_activite'])
df_encoded = pd.DataFrame(category_encoded, columns=category_columns)
# 🔹 Centrer et normaliser les features numériques
scaler = StandardScaler()
features_scaled = scaler.fit_transform(df[['shon','estimation_ges','consommation_energie']])

# Convertir en DataFrame pour manipulation facile
df_features_scaled = pd.DataFrame(features_scaled, columns=['shon','estimation_ges','consommation_energie'])

# Fusionner features normalisés + variables one-hot encodées
df_combined = pd.concat([df_features_scaled, df_encoded], axis=1)


# Étape 2: Utilisation de KNNImputer
imputer = KNNImputer(n_neighbors=3)
df_imputed = pd.DataFrame(imputer.fit_transform(df_combined), columns=df_combined.columns)
df_imputed.head()
# Étape 3: Reconvertir les colonnes One-Hot en catégorie
df_imputed[category_columns] = (df_imputed[category_columns] == df_imputed[category_columns].to_numpy().max(axis=1, keepdims=True)).astype(int)
df_imputed['secteur_activite'] = encoder.inverse_transform(df_imputed[category_columns]).ravel()

# Garder uniquement les colonnes originales
df_final = df[['shon','estimation_ges','consommation_energie']].copy()
df_final['secteur_activite'] = df_imputed['secteur_activite']


#%%  Séparer les bâtiments selon le secteur d'activité


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

