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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Steinschlagrisiko-Challenge (cwm1)
# ## Yannic Lais, Philip Tanner, Marvin von Rappard und Luca Mazzotta

# %%
import pandas as pd
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt
from fitter import Fitter, get_common_distributions, get_distributions

# %%
#Daten einlesen
ablöseZone1 = pd.read_csv("./data/out_1_getrennt.csv", sep=";")
ablöseZone2 = pd.read_csv("./data/out_2_getrennt.csv", sep=";")

print('Ablösezone 1')
display(ablöseZone1.head(5))
print('Ablösezone 2')
display(ablöseZone2.head(5))

# %%
#Datum und Uhrzeit richtig formatieren und in eine Spalte
ablöseZone1["Datum/Zeit"] = pd.to_datetime(ablöseZone1["Datum"] + " " + ablöseZone1["Uhrzeit"], format='%d.%m.%Y %H:%M')
ablöseZone2["Datum/Zeit"] = pd.to_datetime(ablöseZone2["Date"]  + " " + ablöseZone2["Uhrzeit"], format='%d.%m.%Y %H:%M')
#Alte Datum und Uhrzeit Spalten löschen
ablöseZone1 = ablöseZone1.drop(["Datum", "Uhrzeit"], axis = 1)
ablöseZone2 = ablöseZone2.drop(["Date", "Uhrzeit"], axis = 1)
#Nach Datum sortieren, um nacher die Zeitunterschiede besser berechnen zu können
ablöseZone1.sort_values(by=["Datum/Zeit"])
ablöseZone2.sort_values(by=["Datum/Zeit"])

ablöseZone1.head(5)

# %%
#Die Zeit zwischen den Steinschlägen berechnen
#Ablösezone1
unterschiedeZone1 = []
for i in range(len(ablöseZone1)-1):
    unterschied = int(abs(ablöseZone1["Datum/Zeit"][i] - ablöseZone1["Datum/Zeit"][i+1]).total_seconds() / 3600)
    unterschiedeZone1.append(unterschied)
#Aus Array ein pandas-df machen damit man es plotten kann
dfunterschiedeZone1 = pd.DataFrame(unterschiedeZone1, columns=['Unterschiede'])

#Ablösezone2
unterschiedeZone2 = []
for i in range(len(ablöseZone2)-1):
    unterschied = int(abs(ablöseZone2["Datum/Zeit"][i] - ablöseZone2["Datum/Zeit"][i+1]).total_seconds() / 3600)
    unterschiedeZone2.append(unterschied)
#Aus Array ein df machen damit man es plotten kann
dfunterschiedeZone2 = pd.DataFrame(unterschiedeZone2, columns=['Unterschiede'])


# %% [markdown]
# Für die Montecarlosimulation braucht man die Verteilungen mit den dazugehörigen Parameter. Um ein Überblick zu bekommen über die Daten wurden als erstes Histogramme geplottet.

# %% [markdown]
# ### Histogramme Geschwindigkeit

# %%
ablöseZone1.hist(column="v [m/s]", bins=40)
plt.title("Histogramm Geschwindigkeit Ablösezone 1")
plt.ylabel("Anzahl Steine")
plt.xlabel("Gechwindigkeit in m/s")

ablöseZone2.hist(column="v [m/s]", bins=40)
plt.title("Histogramm Geschwindigkeit Ablösezone 2")
plt.ylabel("Anzahl Steine")
plt.xlabel("Gechwindigkeit in m/s")

# %% [markdown]
# ### Histogramme Masse

# %%
ablöseZone1.hist(column="m [kg]", bins=40)
plt.title("Histogramm Masse Ablösezone 1")
plt.ylabel("Anzahl Steine")
plt.xlabel("Masse in kg")

ablöseZone2.hist(column="m [kg]", bins=40)
plt.title("Histogramm Masse Ablösezone 2")
plt.ylabel("Anzahl Steine")
plt.xlabel("Masse in kg")

# %% [markdown]
# ### Histogramme Unterschiede

# %%
dfunterschiedeZone1.hist(column="Unterschiede", bins=40)
plt.title("Histogramm Zeitunterschiede Ablösezone 1")
plt.ylabel("Anzahl Steine")
plt.xlabel("Zeit in Stunden")

dfunterschiedeZone2.hist(column="Unterschiede", bins=40)
plt.title("Histogramm Zeitunterschiede Ablösezone 2")
plt.ylabel("Anzahl Steine")
plt.xlabel("Zeit in Stunden")

# %% [markdown]
# # Fitting

# %% [markdown]
# Anhand der Histogramme und Verteilungsfunktion kann man herausfinden welche Verteilung am besten dazu passt, diese Verteilungen und die jeweiligen Parameter braucht man um die Zufallsvariabeln zu bestimmen für die Simulation.
# Dieses Vorgehen haben wir mit "Scipy" gemacht, wir haben zu erst über die Histogramme mit dem Befehl "get_common_distributions()" bekannte Verteilungen fitten lassen. Mit dem Befehl "get_best" haben wir die Verteilung bekommen die am besten dazu passt. ACHTUNG: Die "get_best" Funktion wurde mit der methode "sumsquare_error" erstellt, somit wird von allen Funktionen die Summe dieses "Square_error" gerechnet. Das heisst, dass die Funktion mit der kleinsten Summe als beste angesehen wird. Dies ist allerdings nicht immer richtig. Deshalb wurden nach dieser "get_best" Funktion die besten drei Verteilungen genommen und diese noch über die Verteilungsfunktion geplottet, so können wir visuell noch schauen welche von den 3 Verteilungen am besten zu der Verteilungsfunktion passt.

# %%
#Liste der common_distributions:
get_common_distributions()

# %% [markdown]
# ### Zone 1, Geschwindigkeit

# %% [markdown]
# #### Über Histogramme gefittet

# %%
ablöseZone1FittedV = Fitter(ablöseZone1["v [m/s]"], distributions=get_common_distributions())
ablöseZone1FittedV.fit()
ablöseZone1FittedV.summary()
ablöseZone1FittedV.get_best(method = 'sumsquare_error')
plt.title("Fitting über PDF-Verteilung, Geschwindigkeit Zone 1")
plt.ylabel("Dichte")
plt.xlabel("Gechwindigkeit in m/s")

# %% [markdown]
# #### Parameter für Top 3 Verteilungen bestimmen

# %%
zone1NormalVerteilungV = ablöseZone1FittedV.fitted_param["norm"]
zone1LognormVerteilungV = ablöseZone1FittedV.fitted_param["lognorm"]
zone1GammaVerteilungV = ablöseZone1FittedV.fitted_param["gamma"]

print(zone1NormalVerteilungV)
print(zone1LognormVerteilungV)
print(zone1GammaVerteilungV)

# %% [markdown]
# #### Top 3 Verteilungen über Verteilungsfunktion plotten

# %%
plt.plot(np.sort(ablöseZone1["v [m/s]"]), np.linspace(0, 1, len(ablöseZone1["v [m/s]"]), endpoint=False), label = "CDF-Verteilung")

plt.plot(np.linspace(1,16,100), scipy.stats.norm.cdf(loc = zone1NormalVerteilungV[0], scale = zone1NormalVerteilungV[1], x = np.linspace(2,16,100)), label = "norm")
plt.plot(np.linspace(1,16,100), scipy.stats.lognorm.cdf(s = zone1LognormVerteilungV[0], loc = zone1LognormVerteilungV[1], scale = zone1LognormVerteilungV[2], x = np.linspace(2,16,100)),label = "lognorm")
plt.plot(np.linspace(1,16,100), scipy.stats.gamma.cdf(a = zone1GammaVerteilungV[0], loc = zone1GammaVerteilungV[1], scale = zone1GammaVerteilungV[2], x = np.linspace(2,16,100)),label = "gamma")
plt.legend(loc="upper left")
plt.title("Fitting über CDF-Verteilung, Geschwindigkeit Zone 1")
plt.ylabel("Dichte")
plt.xlabel("Gechwindigkeit in m/s")


# %% [markdown]
# #### Wir haben uns für "norm" entschieden

# %% [markdown]
# ####  Zufallsvariable bestimmen

# %%
def calcRandomZone1V(num: int):
    return scipy.stats.norm.rvs(loc = zone1NormalVerteilungV[0], scale = zone1NormalVerteilungV[1], size = num)


# %% [markdown]
# ### Zone 2, Geschwindigkeit

# %% [markdown]
# #### Über Histogramme gefittet

# %%
ablöseZone2FittedV = Fitter(ablöseZone2["v [m/s]"], distributions=get_common_distributions())
ablöseZone2FittedV.fit()
ablöseZone2FittedV.summary()
ablöseZone2FittedV.get_best(method = 'sumsquare_error')
plt.title("Fitting über PDF-Verteilung, Geschwindigkeit Zone 2")
plt.ylabel("Dichte")
plt.xlabel("Gechwindigkeit in m/s")

# %% [markdown]
# #### Parameter für Top 3 Verteilungen bestimmen

# %%
zone2PowerlawVerteilungV = ablöseZone2FittedV.fitted_param["powerlaw"]
zone2NormalVerteilungV = ablöseZone2FittedV.fitted_param["norm"]
zone2LognormVerteilungV = ablöseZone2FittedV.fitted_param["lognorm"]

print(zone2PowerlawVerteilungV)
print(zone2NormalVerteilungV)
print(zone2LognormVerteilungV)

# %% [markdown]
# #### Top 3 Verteilungen über Verteilungsfunktion plotten

# %%
plt.plot(np.sort(ablöseZone2["v [m/s]"]), np.linspace(0, 1, len(ablöseZone2["v [m/s]"]), endpoint=False), label = "CDF-Verteilung")
plt.plot(np.linspace(1,55,100), scipy.stats.powerlaw.cdf(a = zone2PowerlawVerteilungV[0], loc = zone2PowerlawVerteilungV[1], scale = zone2PowerlawVerteilungV[2], x = np.linspace(1,55,100)),label = "powerlaw")
plt.plot(np.linspace(1,55,100), scipy.stats.norm.cdf(loc = zone2NormalVerteilungV[0], scale = zone2NormalVerteilungV[1], x = np.linspace(1,55,100)),label = "norm")
plt.plot(np.linspace(1,55,100), scipy.stats.lognorm.cdf(s = zone2LognormVerteilungV[0], loc = zone2LognormVerteilungV[1], scale = zone2LognormVerteilungV[2], x = np.linspace(1,55,100)),label = "lognorm")
plt.legend(loc="upper left")
plt.title("Fitting über CDF-Verteilung, Geschwindigkeit Zone 2")
plt.ylabel("Dichte")
plt.xlabel("Gechwindigkeit in m/s")


# %% [markdown]
# #### Wir haben uns für "norm" entschieden

# %% [markdown]
# ####  Zufallsvariable bestimmen

# %%
def calcRandomZone2V(num: int):
    return scipy.stats.norm.rvs(loc = zone2NormalVerteilungV[0], scale = zone2NormalVerteilungV[1], size = num)


# %% [markdown]
# ### Zone 1, Masse

# %% [markdown]
# #### Über Histogramme gefittet

# %%
ablöseZone1FittedM = Fitter(ablöseZone1["m [kg]"], distributions=get_common_distributions())
ablöseZone1FittedM.fit()
ablöseZone1FittedM.summary()
ablöseZone1FittedM.get_best(method = 'sumsquare_error')
plt.title("Fitting über PDF-Verteilung, Masse Zone 1")
plt.ylabel("Dichte")
plt.xlabel("Masse in Kg.")

# %% [markdown]
# #### Parameter für Top 3 Verteilungen bestimmen

# %%
zone1CauchyVerteilungM = ablöseZone1FittedM.fitted_param["cauchy"]
zone1ExponVerteilungM = ablöseZone1FittedM.fitted_param["expon"]
zone1GammaVerteilungM = ablöseZone1FittedM.fitted_param["gamma"]

print(zone1CauchyVerteilungM)
print(zone1ExponVerteilungM)
print(zone1GammaVerteilungM)

# %% [markdown]
# #### Top 3 Verteilungen über Verteilungsfunktion plotten

# %%
plt.plot(np.sort(ablöseZone1["m [kg]"]), np.linspace(0, 1, len(ablöseZone1["m [kg]"]), endpoint=False), label = "CDF-Verteilung")
plt.plot(np.linspace(1,3500,100), scipy.stats.cauchy.cdf(loc = zone1CauchyVerteilungM[0], scale = zone1CauchyVerteilungM[1], x = np.linspace(1,3500,100)),label = "cauchy")
plt.plot(np.linspace(1,3500,100), scipy.stats.expon.cdf(loc = zone1ExponVerteilungM[0], scale = zone1ExponVerteilungM[1], x = np.linspace(1,3500,100)),label = "expon")
plt.plot(np.linspace(1,3500,100), scipy.stats.gamma.cdf(a = zone1GammaVerteilungM[0], loc = zone1GammaVerteilungM[1], scale = zone1GammaVerteilungM[2], x = np.linspace(1,3500,100)),label = "gamma")
plt.legend(loc="lower right")
plt.title("Fitting über CDF-Verteilung, Masse Zone 1")
plt.ylabel("Dichte")
plt.xlabel("Masse in Kg.")


# %% [markdown]
# #### Wir haben uns für "expon" entschieden

# %% [markdown]
# ####  Zufallsvariable bestimmen

# %%
def calcRandomZone1M(num: int):
    return scipy.stats.expon.rvs(loc = zone1ExponVerteilungM[0], scale = zone1ExponVerteilungM[1], size = num)


# %% [markdown]
# ### Zone 2, Masse

# %% [markdown]
# #### Über Histogramme gefittet

# %%
ablöseZone2FittedM = Fitter(ablöseZone2["m [kg]"], distributions=get_common_distributions())
ablöseZone2FittedM.fit()
ablöseZone2FittedM.summary()
ablöseZone2FittedM.get_best(method = 'sumsquare_error')
plt.title("Fitting über PDF-Verteilung, Masse Zone 2")
plt.ylabel("Dichte")
plt.xlabel("Masse in Kg.")

# %% [markdown]
# #### Parameter für Top 3 Verteilungen bestimmen

# %%
zone2CauchyVerteilungM = ablöseZone2FittedM.fitted_param["cauchy"]
zone2ExponVerteilungM = ablöseZone2FittedM.fitted_param["expon"]
zone2ExponpowVerteilungM = ablöseZone2FittedM.fitted_param["exponpow"]

print(zone2CauchyVerteilungM)
print(zone2ExponVerteilungM)
print(zone2ExponpowVerteilungM)

# %% [markdown]
# #### Top 3 Verteilungen über Verteilungsfunktion plotten

# %%
plt.plot(np.sort(ablöseZone2["m [kg]"]), np.linspace(0, 1, len(ablöseZone2["m [kg]"]), endpoint=False), label = "CDF-Verteilung")
plt.plot(np.linspace(1,450,100), scipy.stats.cauchy.cdf(loc = zone2CauchyVerteilungM[0], scale = zone2CauchyVerteilungM[1], x = np.linspace(1,450,100)),label = "cauchy")
plt.plot(np.linspace(1,450,100), scipy.stats.expon.cdf(loc = zone2ExponVerteilungM[0], scale = zone2ExponVerteilungM[1], x = np.linspace(1,450,100)),label = "expon")
plt.plot(np.linspace(1,450,100), scipy.stats.exponpow.cdf(b = zone2ExponpowVerteilungM[0], loc = zone2ExponpowVerteilungM[1], scale = zone2ExponpowVerteilungM[2], x = np.linspace(1,450,100)),label = "exponpow")
plt.legend(loc="lower right")
plt.title("Fitting über CDF-Verteilung, Masse Zone 2")
plt.ylabel("Dichte")
plt.xlabel("Masse in Kg.")


# %% [markdown]
# #### Wir haben uns für "expon" entschieden

# %% [markdown]
# ####  Zufallsvariable bestimmen

# %%
def calcRandomZone2M(num: int):
    return scipy.stats.expon.rvs(loc = zone2ExponVerteilungM[0], scale = zone2ExponVerteilungM[1], size = num)


# %% [markdown]
# ### Zone 1, Unterschiede

# %% [markdown]
# #### Über Histogramme gefittet

# %%
ablöseZone1FittedU = Fitter(dfunterschiedeZone1["Unterschiede"], distributions=get_common_distributions())
ablöseZone1FittedU.fit()
ablöseZone1FittedU.summary()
ablöseZone1FittedU.get_best(method = 'sumsquare_error')
plt.title("Fitting über PDF-Verteilung, Unterschiede Zone 1")
plt.ylabel("Anzahl Steine")
plt.xlabel("Zeit in Stunden")

# %% [markdown]
# #### Parameter für Top 3 Verteilungen bestimmen

# %%
zone1GammaVerteilungU = ablöseZone1FittedU.fitted_param["gamma"]
zone1PowerlawVerteilungU = ablöseZone1FittedU.fitted_param["powerlaw"]
zone1ExponpowVerteilungU = ablöseZone1FittedU.fitted_param["exponpow"]
zone1ExponVerteilungU = ablöseZone1FittedU.fitted_param["expon"]

print(zone1GammaVerteilungU)
print(zone1PowerlawVerteilungU)
print(zone1ExponpowVerteilungU)
print(zone1ExponVerteilungU)

# %% [markdown]
# #### Top 3 Verteilungen über Verteilungsfunktion plotten

# %%
plt.plot(np.sort(dfunterschiedeZone1["Unterschiede"]), np.linspace(0, 1, len(dfunterschiedeZone1["Unterschiede"]), endpoint=False), label = "CDF-Verteilung")
plt.plot(np.linspace(1,200,100), scipy.stats.gamma.cdf(a = zone1GammaVerteilungU[0], loc = zone1GammaVerteilungU[1], scale = zone1GammaVerteilungU[2], x = np.linspace(1,200,100)),label = "gamma")
plt.plot(np.linspace(1,200,100), scipy.stats.powerlaw.cdf(a = zone1PowerlawVerteilungU[0], loc =  zone1PowerlawVerteilungU[1], scale = zone1PowerlawVerteilungU[2], x = np.linspace(1,200,100)),label = "powerlaw")
plt.plot(np.linspace(1,200,100), scipy.stats.exponpow.cdf(b = zone1ExponpowVerteilungU[0], loc = zone1ExponpowVerteilungU[1], scale = zone1ExponpowVerteilungU[2], x = np.linspace(1,200,100)),label = "exponpow")
plt.plot(np.linspace(1,200,100), scipy.stats.expon.cdf(loc = zone1ExponVerteilungU[0], scale = zone1ExponVerteilungU[1], x = np.linspace(1,200,100)),label = "expon")
plt.legend(loc="lower right")
plt.title("Fitting über CDF-Verteilung, Unterschiede Zone 1")
plt.ylabel("Anzahl Steine")
plt.xlabel("Zeit in Stunden")


# %% [markdown]
# #### Wir haben uns für "expon" entschieden

# %% [markdown]
# ####  Zufallsvariable bestimmen

# %%
def calcRandomZone1U(num: int):
    return scipy.stats.expon.rvs(loc = zone1ExponVerteilungU[0], scale = zone1ExponVerteilungU[1], size = num)


# %% [markdown]
# ### Zone 2, Unterschiede

# %% [markdown]
# #### Über Histogramme gefittet

# %%
ablöseZone2FittedU = Fitter(dfunterschiedeZone2["Unterschiede"], distributions=get_common_distributions())
ablöseZone2FittedU.fit()
ablöseZone2FittedU.summary()
ablöseZone2FittedU.get_best(method = 'sumsquare_error')
plt.title("Fitting über PDF-Verteilung, Unterschiede Zone 2")
plt.ylabel("Anzahl Steine")
plt.xlabel("Zeit in Stunden")

# %% [markdown]
# #### Parameter für Top 3 Verteilungen bestimmen

# %%
zone2GammaVerteilungU = ablöseZone2FittedU.fitted_param["exponpow"]
zone2PowerlawVerteilungU = ablöseZone2FittedU.fitted_param["expon"]
zone2ExponpowVerteilungU = ablöseZone2FittedU.fitted_param["gamma"]

print(zone2GammaVerteilungU)
print(zone2PowerlawVerteilungU)
print(zone2ExponpowVerteilungU)

# %% [markdown]
# #### Top 3 Verteilungen über Verteilungsfunktion plotten

# %%
plt.plot(np.sort(dfunterschiedeZone2["Unterschiede"]), np.linspace(0, 1, len(dfunterschiedeZone2["Unterschiede"]), endpoint=False), label ="CDF-Verteilung")
plt.plot(np.linspace(1,250,100), scipy.stats.exponpow.cdf(b = zone2GammaVerteilungU[0], loc = zone2GammaVerteilungU[1], scale = zone2GammaVerteilungU[2], x = np.linspace(1,250,100)),label ="gamma")
plt.plot(np.linspace(1,250,100), scipy.stats.expon.cdf(loc = zone2PowerlawVerteilungU[0], scale = zone2PowerlawVerteilungU[1], x = np.linspace(1,250,100)),label ="powerlaw")
plt.plot(np.linspace(1,250,100), scipy.stats.gamma.cdf(a = zone2ExponpowVerteilungU[0], loc = zone2ExponpowVerteilungU[1], scale = zone2ExponpowVerteilungU[2], x = np.linspace(1,250,100)),label ="exponpow")
plt.ylabel("Anzahl Steine")
plt.xlabel("Zeit in Stunden")
plt.legend(loc="lower right")
plt.title("Fitting über CDF-Verteilung, Unterschiede Zone 2")


# %% [markdown]
# #### Wir haben uns für "exponpow" entschieden

# %% [markdown]
# ####  Zufallsvariable bestimmen

# %%
def calcRandomZone2U(num: int):
    return scipy.stats.exponpow.rvs(b = zone2GammaVerteilungU[0], loc = zone2GammaVerteilungU[1], scale = zone2GammaVerteilungU[2], size = num)


# %% [markdown]
# # Monte-Carlo Simulation

# %%
#Monte Carlo Simulation
def monteCarlo(simulationslängeJahre: int): 
    simulationslänge = simulationslängeJahre * 24 * 365
    
    plan = []
    netzReisser = 0
    zeitReisser = []
    steineImNetz = 0
    stunde = 0
    steine = 0
    i = 0
    laengeliste = simulationslängeJahre * 400

    randomUList1 = calcRandomZone1U(laengeliste)
    randomMList1 = calcRandomZone1M(laengeliste)
    randomVList1 = calcRandomZone1V(laengeliste)

    randomUList2 = calcRandomZone2U(laengeliste)
    randomMList2 = calcRandomZone2M(laengeliste)
    randomVList2 = calcRandomZone2V(laengeliste)

    for j in range(laengeliste-1):
        randomUList1[j+1] = randomUList1[j] + randomUList1[j+1]
        plan.append([randomUList1[j],randomMList1[j],randomVList1[j]])
        randomUList2[j+1] = randomUList2[j] + randomUList2[j+1]
        plan.append([randomUList2[j],randomMList2[j],randomVList2[j]])

    plan.sort()

    while simulationslänge > 0:
        i += 1

        zeitUnterschied = plan[i][0] - plan[i-1][0]
        simulationslänge -= zeitUnterschied
        stunde += zeitUnterschied

        randomM = plan[i][1]
        randomV = plan[i][2]

        potentielleEnergie = (0.5 * randomM * (randomV**2)) / 1000
        steine += 1

        #Netz reisst sicher
        if potentielleEnergie > 1000 \
            or zeitUnterschied < 24 and steineImNetz > 2000 and potentielleEnergie > 500:
            netzReisser += 1
            zeitReisser.append(stunde)
            steineImNetz = 0
        #Netz reisst nicht, Netz ist geräumt
        elif zeitUnterschied > 24 and potentielleEnergie < 1000:
            steineImNetz = randomM
        #Netz reisst nicht --> 
        # im Netz sind > 2000 kg --> poteng < 500kJ:
        # Netz sind < 2000 kg --> poteng < 1000kJ:
        elif zeitUnterschied < 24 and steineImNetz > 2000 and potentielleEnergie < 500 \
            or zeitUnterschied < 24 and steineImNetz < 2000 and potentielleEnergie < 1000:
            steineImNetz += randomM
    
    return (steine, netzReisser, zeitReisser)


# %%
jahre = 500_000
resultat = monteCarlo(jahre)
print("Anzahl Steinschläge:", resultat[0])
print("Anzahl Netzreisser:", resultat[1])
print("Die ersten 20 Zeitpunkte wann das Netz gerissen ist (In Stunden):", resultat[2][:20])

# %%
#Kontrollieren ob die Netzreisser eine Abhängigkeit haben mit der Uhrzeit haben
uhrzeiten = []
for i in range(len(resultat[2])):
    uhrzeiten.append(resultat[2][i]%24)
    
dfUhrzeiten = pd.DataFrame(uhrzeiten, columns=['uhrzeit'])
dfUhrzeiten.hist(column="uhrzeit", bins=24)

# %% [markdown]
# Wie man anhand des Histogramms erkennen kann, besteht kein Zusammenhang zwischen wann das Netz gerissen ist und die Uhrzeit. Somit spielt die Verteilung der Autos keine Rolle. Deshalb werden die 1200 Auto gleichmässig über den Tag verteilt.

# %%
zeitpunkteAuto = []
auto = 0
while auto < 24:
    zeitpunkteAuto.append(auto)
    auto = round(auto + 0.02, 2)

# %% [markdown]
# Annahme Strasse (Spielt keine eigentlich keine Rolle für die Berechnung, jedoch kann man es sich so besser vorstellen): <br>
# Strassenlänge 200m: <br>
# Anfang = 0m <br>
# Ablösezone 1 = 100m <br>
# Ablösezone 2 = 100m (Höher als Ablösezone 1) <br>
# Ende = 200m

# %% [markdown]
# Annahme: Wenn man mit 60 km/h (16.67 m/s) gegen ein Stein fährt stirbt man mit einer Wahrscheinlichkeit von 5% (Siehe Bericht) <br>
# Reaktionszeit (Vorbremszeit) = 1sek (Quelle: https://de.wikipedia.org/wiki/Reaktion_(Verkehrsgeschehen)) <br>
# <br>
# Quelle für Bremszeit und Bremsweg: <br>
# https://www.johannes-strommer.com/formeln/weg-geschwindigkeit-beschleunigung-zeit/ <br>
# Bremsweg = 1sek * 16.67m/s = 16.67m <br>

# %% [markdown]
# Da wir für beide Ablösezonen eine Liste haben mit dem Zeitpunkt (In Stunden) wann die Netze gerissen sagen, können wir 2 Aussagen betätigen: <br>
# - Befindet sich das Auto über Meter 83.33 und unter Meter 100 im Zeitintervall [(ZeitpunktpunktreisserZone1 - 1sek) bis (ZeitpunktreisserZone1)] schafft das Auto es nicht mehr zu bremsen und somit wird es zu mit einer Wahrscheinlichkeit von 5% ein Todesfall.
# - Befindet sich das Auto genau unter dem Stein (Todeszone = 2m) wird es zu 100% ein Todesfall.

# %%
startGefahrenzone = 83.33
geschwindigkeit = 16.67


# %%
def toHours(seconds:float) -> float:
    return seconds/60/60


# %%
tote = 0
ZeitBisGefahrenzone = toHours(startGefahrenzone/geschwindigkeit)
todesWahrscheinlichkeitMit60 = 0.05
durchschnittlichePersonenImAuto = 1.56
for uhrzeit in uhrzeiten:
    for zeitpunkt in zeitpunkteAuto:
        ZeitStartGefahrenzone = zeitpunkt + ZeitBisGefahrenzone
        if ZeitStartGefahrenzone > uhrzeit - toHours(1) and \
            ZeitStartGefahrenzone < uhrzeit - toHours(0.12): #Auto schafft es nicht zu bremsen
            tote += (1 * todesWahrscheinlichkeitMit60 * durchschnittlichePersonenImAuto)
        elif ZeitStartGefahrenzone >= uhrzeit - toHours(0.12) and \
            ZeitStartGefahrenzone <= uhrzeit: #Stein fällt auf Auto
            tote += 1 * durchschnittlichePersonenImAuto

# %%
print("Anzahl Tote:", tote)
print("Wahrscheinlichkeit das jemand stirbt:", tote / jahre)
print("Unsere Wahrscheinlichkeit ist höher als der Grenzwert:",  tote / jahre > 10**-4)

# %% [markdown]
# ## Fazit

# %% [markdown]
# ### Da unsere Wahrscheinlichkeit grösser ist als der Grenzwert, muss die Strasse gesperrt werden.
