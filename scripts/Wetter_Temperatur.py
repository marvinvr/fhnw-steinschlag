# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3.10.0 64-bit
#     name: python3
# ---

# %% [markdown]
# # Korrelation Steinschlag - Wetter

# %% [markdown]
# ### Imports

# %%
import pandas as pd
from matplotlib import pyplot as plt

# %% [markdown]
# ### Daten Importieren

# %%
zone_1 = pd.read_csv('./data/out_1.csv', ';')
zone_2 = pd.read_csv('./data/out_2.csv', ';')
weather = pd.read_csv('./data/weather.csv')

# %%
display(zone_1.head())
display(zone_2.head())
display(weather.head())

# %% [markdown]
# ## Daten Aufbereiten

# %%
zone_1['Date'] = pd.to_datetime(zone_1['Datum'], format='%d.%m.%Y')
zone_2['Date'] = pd.to_datetime(zone_2['Date'], format='%d.%m.%Y')
weather['Date'] = pd.to_datetime(weather['date'], format='%Y-%m-%d')

weather['tdif'] = weather['tmax'] - weather['tmin']

idx = pd.date_range('2019-01-01', '2019-03-30')

combined_zones = pd.concat([zone_1, zone_2])
combined_zones_count = combined_zones['Date'].value_counts().reindex(idx, fill_value=0)

# %% [markdown]
# ## Visualisierungen

# %% [markdown]
# ### Durchschnittliche Temperatur über die drei Monaten

# %%
_, ax = plt.subplots()
ax2 = ax.twiny()
ax2.axes.get_xaxis().set_visible(False)
weather.set_index('Date')['tavg'].plot(ylabel='Grad Celsius', xlabel='Average Temperature', ax=ax)
combined_zones_count.plot(color='red', xlabel='Anzahl Steinschläge', ax=ax2)
plt.title('Totale Anzahl Steinschläge Verglichen mit Temperatur')
plt.legend()
plt.show()

# %% [markdown]
# Im obenstehenden plot sieht man die Temperatur in Kombination mit der Anzahl Steinschläge. Hier ist keine grosse korellation zu sehen, die Temperatur ansich scheint keinen grossen Einfluss zu haben.

# %%
_, ax = plt.subplots()
ax2 = ax.twiny()
ax2.axes.get_xaxis().set_visible(False)
weather.set_index('Date')['tdif'].plot(ylabel='Average Temperature (Celsius)', xlabel='Date', ax=ax)
combined_zones_count.plot(color='red', ax=ax2)
plt.title('Totale Anzahl Steinschläge Verglichen mit Der Termperaturschwankung am respektiven Tag')
plt.show()

# %% [markdown]
# Zusätzlich zum oberen Diagramm hier nicht die anzahl Steinschläge im Vergleich zu der Temperaturschwankung am respektiven Tag. Auch hier ist keine direkte Korrelation zu sehen.

# %%
_, ax = plt.subplots()
ax2 = ax.twiny()
ax3 = ax2.twiny()
ax2.axes.get_xaxis().set_visible(False)
ax3.axes.get_xaxis().set_visible(False)
weather.set_index('Date')['prcp'].plot( ax=ax)
combined_zones_count.plot(color='red', ax=ax2, xlabel='Anzahl Steinschläge')
plt.title('Totale Anzahl Steinschläge Verglichen mit Regenaufkommen')
plt.show()

# %% [markdown]
# Auch beim Regenfall sehen wir keinen direkten Einfluss auf die Anzahl Steinschläge.

# %%
_, ax = plt.subplots()
ax2 = ax.twiny()
ax2.axes.get_xaxis().set_visible(False)
(weather.set_index('Date')['snow'] / 50).plot( ax=ax)
combined_zones_count.plot(color='red', ax=ax2, xlabel='Anzahl Steinschläge')
plt.title('Totale Anzahl Steinschläge Verglichen mit Schneefall')
plt.show()

# %% [markdown]
# Beim Vergleich mit dem Schneefall sehen wir eine Leichte Steigung der Anzahl Steinschläge, jedoch nicht genug um auf eine Eindeutige Korrelation hinzudeuten.

# %% [markdown]
# Wir gehen daher davon aus dass das Wetter keinen Einfluss auf die Anzahl Steinschläge hat.

# %%
_, ax = plt.subplots()
ax2 = ax.twiny()
ax2.axes.get_xaxis().set_visible(False)
combined_zones.set_index('Date')['m [kg]'].plot(ax=ax2, kind='bar')
weather.set_index('Date')['snow'].plot(color='red', ax=ax, xlabel='Anzahl Steinschläge')
plt.title('Gewicht von Steinschlägen Verglichen mit Schneefall')
plt.show()


# %% [markdown]
# Was uns jedoch hier auffällt ist dass Schnee einen Einfluss auf das Gewicht der Steinschläge hat. 

# %%
