import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Charger les données
datasets = {
    "Centre Commercial": "centre_commercial_mois.csv",
    "Occupation Continue": "occupation_continue_mois.csv",
    "Autre": "autre_mois.csv",
    "Bureau/Admin/Enseignement": "bureau_admin_enseignement_mois.csv"
}

# Initialisation des résultats
scores = {}

plt.figure(figsize=(12, 10))
colors = ["blue", "blue", "blue", "blue"]

for i, (name, file) in enumerate(datasets.items(), 1):
    df = pd.read_csv(file)

    # Renommage des colonnes
    df.rename(columns={"mois": "ds", "estimation_ges": "y"}, inplace=True)
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df.sort_values(by="ds")

    # Découpage Train/Test
    train_size = int(len(df) * 0.8)
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]

    # Modèle Prophet (sans saisonnalité)
    model = Prophet()
    model.fit(train)
    
    # Prédictions
    future = test[["ds"]]
    forecast = model.predict(future)

    # Calcul des scores
    y_true = test["y"].values
    y_pred = forecast["yhat"].values
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # En pourcentage
    
    scores[name] = {"MAE": mae, "RMSE": rmse, "MAPE": mape}

    # Affichage des résultats
    plt.subplot(2, 2, i)
    plt.plot(train["ds"], train["y"], label="Train", color="black")
    plt.plot(test["ds"], test["y"], label="Test (réel)", color=colors[i-1])
    plt.plot(test["ds"], forecast["yhat"], label="Prédictions", color="red")
    plt.fill_between(test["ds"], forecast["yhat_lower"], forecast["yhat_upper"], color='pink', alpha=0.3)

    plt.title(f"{name}")
    plt.legend()
    plt.ylim(0, 100)

plt.tight_layout()
plt.show()

# Afficher les scores
for domain, metrics in scores.items():
    print(f"🔹 Scores pour {domain} :")
    print(f"   📌 MAE  = {metrics['MAE']:.2f}")
    print(f"   📌 RMSE = {metrics['RMSE']:.2f}")
    print(f"   📌 MAPE = {metrics['MAPE']:.2f}%\n")
