import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Liste des fichiers CSV et des secteurs correspondants
# files = {
#     'occupation_continue.csv': 'Occupation Continue',
#     'centre_commercial.csv': 'Centre Commercial',
#     'autre.csv': 'Autre',
#     'bureau_admin_enseignement.csv': 'Bureau Admin Enseignement'
# }
import matplotlib.pyplot as plt

# Liste des fichiers CSV et des secteurs correspondants
# files = {
#     'occupation_continue.csv': 'Occupation Continue',
#     'centre_commercial.csv': 'Centre Commercial',
#     'autre.csv': 'Autre',
#     'bureau_admin_enseignement.csv': 'Bureau Admin Enseignement'
# }

files = {
    'occupation_continue_mois.csv': 'Occupation Continue',
    'centre_commercial_mois.csv': 'Centre Commercial',
    'autre_mois.csv': 'Autre',
    'bureau_admin_enseignement_mois.csv': 'Bureau Admin Enseignement'
}

# fig, axs = plt.subplots(len(files), 1, figsize=(12, 8), sharex=True)

# for i, (file, sector) in enumerate(files.items()):
#     # Lire le fichier CSV
#     df = pd.read_csv(file)
    
#     # Convertir la colonne 'date_etablissement_dpe' en datetime
#     # df['date_etablissement_dpe'] = pd.to_datetime(df['date_etablissement_dpe'], format ='mixed')
#     # df.set_index('date_etablissement_dpe', inplace=True)
    
#     df['mois'] = pd.to_datetime(df['mois'], format ='mixed')
#     df.set_index('mois', inplace=True)

#     # Split data into training (80%) and testing (20%)
#     train_size = int(len(df) * 0.8)
#     train, test = df.iloc[:train_size], df.iloc[train_size:]
    
#     # Ajuster le modèle ARIMA
#     model = ARIMA(train['estimation_ges'], order=(10, 1, 10))
#     model_fit = model.fit()
    
#     # Résumé du modèle
#     print(f"Résumé du modèle pour le secteur {sector}:")
#     print(model_fit.summary())
    
#     # Faire des prédictions sur les données d'entraînement
#     train_predictions = model_fit.predict(start=train.index[0], end=train.index[-1])
    
#     # Faire des prédictions sur les données de test
#     forecast = model_fit.forecast(steps=len(test))

#     mape = mean_absolute_percentage_error(test['estimation_ges'], forecast)
#     print(f"MAPE pour le secteur {sector}: {mape:.2f}")
    
#     # Faire des prédictions jusqu'en 2030
#     future_forecast = model_fit.forecast(steps=(2030 - df.index[-1].year) * 12)

#     # Tracer les résultats
#     axs[i].plot(df['estimation_ges'], label=f'Historique {sector}')
#     axs[i].plot(train.index, train_predictions, label=f'Prévisions Entraînement {sector}', color='red')
#     axs[i].plot(test.index, test['estimation_ges'], label=f'Réalité {sector}', color='green')
#     axs[i].plot(test.index, forecast, label=f'Prévisions {sector}', linestyle='--')
#     axs[i].plot(pd.date_range(start=test.index[-1], periods=len(future_forecast), freq='M'), future_forecast, label=f'Prévisions Futur {sector}', linestyle='--', color='purple')
#     axs[i].set_title(f'Secteur {sector}')
#     axs[i].legend()

# plt.xlabel('Date')
# plt.ylabel('Emission GES')
# plt.suptitle('Prévisions des émissions de GES avec ARIMA pour chaque secteur d\'activité')
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()
# files = {
#     'occupation_continue_mois.csv': 'Occupation Continue',
#     'centre_commercial_mois.csv': 'Centre Commercial',
#     'autre_mois.csv': 'Autre',
#     'bureau_admin_enseignement_mois.csv': 'Bureau Admin Enseignement'
# }

fig, axs = plt.subplots(len(files), 1, figsize=(12, 8), sharex=True)

for i, (file, sector) in enumerate(files.items()):
    # Lire le fichier CSV
    df = pd.read_csv(file)
    
    # Convertir la colonne 'date_etablissement_dpe' en datetime
    # df['date_etablissement_dpe'] = pd.to_datetime(df['date_etablissement_dpe'], format ='mixed')
    # df.set_index('date_etablissement_dpe', inplace=True)
    
    df['mois'] = pd.to_datetime(df['mois'], format ='mixed')
    df.set_index('mois', inplace=True)

    # Split data into training (80%) and testing (20%)
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    
    # Ajuster le modèle ARIMA
    model = ARIMA(train['estimation_ges'], order=(10, 1, 10))
    model_fit = model.fit()
    
    # Résumé du modèle
    print(f"Résumé du modèle pour le secteur {sector}:")
    print(model_fit.summary())
    
    # Faire des prédictions sur les données d'entraînement
    train_predictions = model_fit.predict(start=train.index[0], end=train.index[-1])
    
    # Faire des prédictions sur les données de test
    forecast = model_fit.forecast(steps=len(test))

    mape = mean_absolute_percentage_error(test['estimation_ges'], forecast)
    print(f"MAPE pour le secteur {sector}: {mape:.2f}")
    
    # Tracer les résultats
    axs[i].plot(df['estimation_ges'], label=f'Historique {sector}')
    axs[i].plot(train.index, train_predictions, label=f'Prévisions Entraînement {sector}', color='red')
    axs[i].plot(test.index, test['estimation_ges'], label=f'Réalité {sector}', color='green')
    axs[i].plot(test.index, forecast, label=f'Prévisions {sector}', linestyle='--')
    axs[i].set_title(f'Secteur {sector}')
    axs[i].legend()

plt.xlabel('Date')
plt.ylabel('Emission GES')
plt.suptitle('Prévisions des émissions de GES avec ARIMA pour chaque secteur d\'activité')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
