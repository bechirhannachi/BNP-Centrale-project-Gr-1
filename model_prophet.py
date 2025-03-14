import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Charger les données
df = pd.read_csv("data1_agg.csv")

# Renommer les colonnes pour Prophet
df.rename(columns={"date_etablissement_dpe": "ds", "moyenne_ponderee_ges": "y"}, inplace=True)

# Convertir la colonne ds en datetime
df["ds"] = pd.to_datetime(df["ds"])

# Trier les données par date (au cas où)
df = df.sort_values(by="ds")

# Découper en 80% train, 20% test
train_size = int(len(df) * 0.8)
train = df.iloc[:train_size]
test = df.iloc[train_size:]

# Initialiser et entraîner le modèle sur train
model = Prophet()
model.fit(train)

# Faire des prédictions sur la période du test
future = test[["ds"]]  # On garde les dates du test
forecast = model.predict(future)

# Visualisation des résultats
plt.figure(figsize=(12, 6))
plt.plot(train["ds"], train["y"], label="Train", color="blue")
plt.plot(test["ds"], test["y"], label="Test (réel)", color="green")
plt.plot(test["ds"], forecast["yhat"], label="Prédictions", color="red")
plt.fill_between(test["ds"], forecast["yhat_lower"], forecast["yhat_upper"], color='pink', alpha=0.3) # Intervalle de confiance
plt.legend()
plt.ylim(0,5000)
plt.title("Comparaison des prédictions et des données réelles")
plt.show()
