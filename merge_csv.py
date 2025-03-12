import pandas as pd

df1 = pd.read_csv("df1.csv", low_memory=False, dtype=str)  # Lire toutes les colonnes en str pour éviter le problème
df2 = pd.read_csv("df2.csv", low_memory=False, dtype=str)




from pyproj import Transformer
transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
df1[['X_L93', 'Y_L93']] = df1.apply(lambda row: transformer.transform(row['longitude'], row['latitude']), axis=1, result_type='expand')

colonnes_1 = list(set(df1.columns))
colonnes_2 = list(set(df2.columns))

correspondances = {
    "surface_utile": "Surface_utile",
    "shon": "Surface_(SHON)",
    "date_etablissement_dpe" : "Date_établissement_DPE",
    #utiliser date_etabablissement_dpe
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

colonnes_numeriques = ["surface_utile", "shon", "estimation_ges", "consommation_energie", "X_L93", "Y_L93"]
for col in colonnes_numeriques:
    df1[col] = pd.to_numeric(df1[col], errors="coerce")
    df2[col] = pd.to_numeric(df2[col], errors="coerce")

# Convertir les colonnes en types compatibles
# for col in colonnes_finales:
#     df1[col] = pd.to_numeric(df1[col], errors="coerce") if pd.api.types.is_numeric_dtype(df2[col]) else df1[col].astype(str)
#     df2[col] = pd.to_numeric(df2[col], errors="coerce") if pd.api.types.is_numeric_dtype(df1[col]) else df2[col].astype(str)



# Concaténer les DataFrames verticalement
df_merged = pd.concat([df1, df2], ignore_index=True)

# Filtrer les valeurs aberrantes
df_merged = df_merged[df_merged["estimation_ges"] >= 0]
df_merged = df_merged[df_merged["surface_utile"] >= 0]
df_merged = df_merged[df_merged["shon"] >= 0]
df_merged = df_merged[df_merged["consommation_energie"] >= 0]


# Sauvegarder le fichier final
df_merged.to_csv("fusion.csv", index=False)


colonnes_finales = list(correspondances.keys())

df_merged = df_merged[colonnes_finales]

print(df_merged.head())
#df_merged.to_csv("fusion.csv", index=False)