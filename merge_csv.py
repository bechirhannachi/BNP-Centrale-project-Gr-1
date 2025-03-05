import pandas as pd

df1 = pd.read_csv("dpe1.csv")
df2 = pd.read_csv("dpe2.csv")


colonnes_1 = list(set(df1.columns))
colonnes_2 = list(set(df2.columns))

correspondances = {
    "surface_utile": "Surface_utile",
    "shon": "Surface_(SHON)",
    "date_visite_diagnostiqueur" : "Date_visite_diagnostiqueur",
    "annee_construction": "Année_construction",
    "secteur_activite": "Secteur_activité",
    #"code_postal": "Code_postal_(brut)",  # il faut voir si c pas mieux de mettre code postal ban
    "classe_estimation_ges" : "Etiquette_GES",
    "estimation_ges" : "Emission_GES_kgCO2/m²/an",
    "consommation_energie" : "Conso_kWhep/m²/an"
}



for col1, col2 in correspondances.items():
    if col1 in df1.columns and col2 in df2.columns:
        type1, type2 = df1[col1].dtype, df2[col2].dtype
        
        if pd.api.types.is_numeric_dtype(df1[col1]) and not pd.api.types.is_numeric_dtype(df2[col2]):
            df2[col2] = pd.to_numeric(df2[col2], errors="coerce")
        elif pd.api.types.is_numeric_dtype(df2[col2]) and not pd.api.types.is_numeric_dtype(df1[col1]):
            df1[col1] = pd.to_numeric(df1[col1], errors="coerce")

for col1, col2 in correspondances.items():
    if col1 in df1.columns and col2 in df2.columns:
        type1, type2 = df1[col1].dtype, df2[col2].dtype
        
        if pd.api.types.is_numeric_dtype(df1[col1]) and not pd.api.types.is_numeric_dtype(df2[col2]):
            df2[col2] = pd.to_numeric(df2[col2], errors="coerce")
        elif pd.api.types.is_numeric_dtype(df2[col2]) and not pd.api.types.is_numeric_dtype(df1[col1]):
            df1[col1] = pd.to_numeric(df1[col1], errors="coerce")

print("Clés de df1 utilisées pour le merge :", list(correspondances.keys()))
print("Clés correspondantes dans df2 :", list(correspondances.values()))
print("Nombre de clés communes :", len(correspondances))

df_merged = pd.merge(
    df1,
    df2,
    left_on=list(correspondances.keys()),
    right_on=list(correspondances.values()),
    how="outer"
)
print(df_merged.head())
#df_merged.to_csv("fusion.csv", index=False)