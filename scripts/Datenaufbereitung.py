# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
Zone1 = pd.read_csv("data/out_1.csv", sep=",")
Zone2 = pd.read_csv("data/out_2.csv", sep=",")

# %%
Zone1['Datum'] = pd.to_datetime(Zone1['Datum'], dayfirst=True)
Zone2['Date'] = pd.to_datetime(Zone2['Date'], dayfirst=True)

# %%
print(Zone1)

# %%
print(Zone2)

# %% [markdown]
# # Häufigkeitsverteilung der Geschwindigkeit

# %% [markdown]
# ## Zone1

# %%
Zone1.hist(column='v [m/s]', bins=20, figsize=(20,10))

# %% [markdown]
# ## Zone2

# %%
Zone2.hist(column='v [m/s]', bins=20, figsize=(20,10))

# %% [markdown]
# # Häufigkeitsverteilung der Masse

# %% [markdown]
# ## Zone1

# %%
Zone1.hist(column='m [kg]', bins=20, figsize=(20,10))

# %% [markdown]
# ## Zone2

# %%
Zone2.hist(column='m [kg]', bins=20, figsize=(20,10))

# %% [markdown]
# # Geschwindigkeit in Abhängigkeit von der Masse

# %% [markdown]
# ## Zone1

# %%
Zone1.plot.scatter(x = 'm [kg]', y = 'v [m/s]', figsize=(15,10))

# %% [markdown]
# ## Zone2

# %%
Zone2.plot.scatter(x = 'm [kg]', y = 'v [m/s]', figsize=(15,10))

# %% [markdown]
# # Steinschläge pro Tag (Tagen an denen nichts passiert ist einfügen)

# %% [markdown]
# ## Zone1

# %%
Zone1['Month'] = Zone1['Datum'].apply(lambda x: "%d" % (x.month))
Zone1
Zone1.groupby('Month').size()

# %%
Zone1.value_counts('Datum').sort_index().plot(kind='bar',figsize=(20,5))

# %% [markdown]
# ## Zone2

# %%
Zone2['Month'] = Zone2['Date'].apply(lambda x: "%d" % (x.month))
Zone2
Zone2.groupby('Month').size()

# %%
Zone2.value_counts('Date').sort_index().plot(kind='bar',figsize=(20,5))

# %% [markdown]
# # Steinschläge pro Stunde

# %% [markdown]
# ## Zone1

# %%
Uhrzeit1 = Zone1.sort_values(by=['Uhrzeit'])
Uhrzeit1.plot.scatter(x = 'Uhrzeit', y = 'm [kg]', figsize=(20,10))

# %%
Zone1.value_counts('Uhrzeit').sort_index().plot(kind='bar',figsize=(20,5))

# %% [markdown]
# ## Zone2

# %%
Uhrzeit2 = Zone2.sort_values(by=['Uhrzeit'])
Uhrzeit2.plot.scatter(x = 'Uhrzeit', y = 'm [kg]', figsize=(20,10))

# %%
Zone2.value_counts('Uhrzeit').sort_index().plot(kind='bar',figsize=(20,5))
