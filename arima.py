import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

import matplotlib.pyplot as plt

df=pd.read_csv('fusion.csv')

# Convertir la colonne 'date' en datetime
df['date_etablissement_dpe'] = pd.to_datetime(df['date_etablissement_dpe'])
df.sort_values(by=['date_etablissement_dpe','secteur_activite'])

filter1=df['shon']>0
filter2=df['surface_utile']>0
filter3=df['estimation_ges']>0
df=df[filter1&filter2&filter3]

Q1=df['estimation_ges'].quantile(0.25)
Q3=df['estimation_ges'].quantile(0.75)
IQR=Q3-Q1
filter=(df['estimation_ges']>=(Q1-1.5*IQR))&(df['estimation_ges']<=(Q3+1.5*IQR))
df=df[filter]

# Calcul du GES total pondéré par la surface SHON
df['ges_weighted'] = df['estimation_ges'] * df['shon']

# Regroupement et calcul du ratio correct
df_grouped = df.groupby(['date_etablissement_dpe','secteur_activite']).agg({
    'ges_weighted': 'sum',  # Somme des GES pondérés
    'shon': 'sum'           # Somme des surfaces utiles
}).reset_index()

# Calcul du GES moyen pondéré
df_grouped['ges_final'] = df_grouped['ges_weighted'] / df_grouped['shon']

print(df_grouped.columns)

# Ajuster le modèle ARIMA pour chaque secteur d'activité
sectors = df_grouped['secteur_activite'].unique()
forecast_steps = 250  # Nombre de périodes à prédire

fig, axs = plt.subplots(len(sectors), 1, figsize=(12, 8), sharex=True)

for i, sector in enumerate(sectors):
    sector_df = df_grouped[df_grouped['secteur_activite'] == sector]
    sector_df.set_index('date_etablissement_dpe', inplace=True)
    
    # Ajuster le modèle ARIMA
    model = ARIMA(sector_df['ges_final'], order =(10,0,10))
    model_fit = model.fit()
    
    # Résumé du modèle
    print(f"Résumé du modèle pour le secteur {sector}:")
    print(model_fit.summary())
    
    # Faire des prédictions
    forecast = model_fit.forecast(steps=forecast_steps)
    
    # Tracer les résultats
    axs[i].plot(sector_df['ges_final'], label=f'Historique {sector}')
    axs[i].plot(pd.date_range(start=sector_df.index[-1], periods=forecast_steps), forecast, label=f'Prévisions {sector}', linestyle='--')
    axs[i].set_title(f'Secteur {sector}')
    axs[i].legend()

plt.xlabel('Date')
plt.ylabel('Emission GES')
plt.suptitle('Prévisions des émissions de GES avec ARIMA pour chaque secteur d\'activité')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()