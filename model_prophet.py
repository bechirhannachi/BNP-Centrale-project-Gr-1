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
    model = Prophet(yearly_seasonality=True,weekly_seasonality=True)
    model.fit(train)
    
    # Prédictions sur la période de test
    future_test = test[["ds"]]
    forecast_test = model.predict(future_test)

    # Calcul des scores
    y_true = test["y"].values
    y_pred = forecast_test["yhat"].values
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # En pourcentage
    mse = mean_squared_error(y_true, y_pred)
    scores[name] = {"MAE": mae, "RMSE": rmse, "MSE": mse, "MAPE": mape}
    # Prédictions jusqu'en 2030
    future_2030 = model.make_future_dataframe(periods=(2030 - df["ds"].dt.year.max()) * 12, freq="M")
    forecast_2030 = model.predict(future_2030)
    forecast_2030 = forecast_2030[forecast_2030["ds"] > test["ds"].max()]
    # Affichage des résultats
    plt.subplot(2, 2, i)
    plt.plot(train["ds"], train["y"], label="Train", color="black")
    plt.plot(test["ds"], test["y"], label="Test (réel)", color=colors[i-1])
    plt.plot(test["ds"], forecast_test["yhat"], label="Prédictions Test", color="red")
    plt.fill_between(test["ds"], forecast_test["yhat_lower"], forecast_test["yhat_upper"], color='pink', alpha=0.3)
    
    # Ajout des prédictions jusqu'en 2030
    plt.plot(forecast_2030["ds"], forecast_2030["yhat"], label="Prédictions 2030", color="green")
    plt.fill_between(forecast_2030["ds"], forecast_2030["yhat_lower"], forecast_2030["yhat_upper"], color='lightgreen', alpha=0.3)

    plt.title(f"{name}\nMSE: {mse:.2f}")
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
    print(f"   📌 MSE  = {metrics['MSE']:.2f}")