import pandas as pd

# Liste des fichiers à traiter
files = ["data1.csv", "data2.csv", "data3.csv", "data4.csv"]

for file in files:
    # Charger les données
    data = pd.read_csv(file)

    # Convertir la colonne date en datetime
    data["date_etablissement_dpe"] = pd.to_datetime(data["date_etablissement_dpe"])

    # Trier les données par date
    data = data.sort_values("date_etablissement_dpe").reset_index(drop=True)

    # Créer une colonne indexée à partir de la plus vieille date
    data["jour_index"] = (data["date_etablissement_dpe"] - data["date_etablissement_dpe"].min()).dt.days

    # Calculer la moyenne pondérée par jour
    result = (
        data.groupby("jour_index", group_keys=False)
        .filter(lambda g: g["surface_utile"].sum() > 0)  # 🔹 On ignore les jours avec surface_utile = 0
        .groupby("jour_index", group_keys=False)
        .apply(lambda g: pd.Series({
            "moyenne_ponderee_ges": (g["estimation_ges"] * g["surface_utile"]).sum() / g["surface_utile"].sum()
        }), include_groups=False)  # 🔹 Supprime le DeprecationWarning
        .reset_index()
    )

    # Sauvegarder dans un nouveau CSV
    result.to_csv(file.replace(".csv", "_agg.csv"), index=False)
