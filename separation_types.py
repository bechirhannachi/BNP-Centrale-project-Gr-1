import pandas as pd

# Charger le fichier fusionné
df = pd.read_csv("fusion.csv")

# Dictionnaire des valeurs possibles et des noms de fichiers associés
dataframes = {
    "Centre commercial": "data1.csv",
    "Autres cas (par exemple: théâtres, salles de sport, restauration, commerces individuels, etc)": "data2.csv",
    "Bâtiment à occupation continue (par exemple: hopitaux, hôtels, internats, maisons de retraite, etc)": "data3.csv",
    "Bâtiment à usage principale de bureau, d'administration ou d'enseignement": "data4.csv"
}

for valeur, fichier in dataframes.items():
    df_filtre = df[df["secteur_activite"] == valeur]
    df_filtre.to_csv(fichier, index=False)
    print(f"{len(df_filtre)} lignes enregistrées dans {fichier}")

