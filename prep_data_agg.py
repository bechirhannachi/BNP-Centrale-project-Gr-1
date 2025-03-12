import pandas as pd

# Charger les données
data = pd.read_csv("data1.csv")

# Convertir la colonne date en datetime
data["date_etablissement_dpe"] = pd.to_datetime(data["date_etablissement_dpe"])

# Trier les données par date
data = data.sort_values("date_etablissement_dpe").reset_index(drop=True)

# Créer une colonne indexée à partir de la plus vieille date
data["jour_index"] = (data["date_etablissement_dpe"] - data["date_etablissement_dpe"].min()).dt.days

# Calculer la moyenne pondérée par jour
result = data.groupby("jour_index").apply(
    lambda g: pd.Series({
        "moyenne_ponderee_ges": (g["estimation_ges"] * g["surface_utile"]).sum() / g["surface_utile"].sum()
    })
).reset_index()

# Sauvegarder dans un nouveau CSV
result.to_csv("data1_agg.csv", index=False)
