import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from houses import houses_copy, num_features


# wykres macierzy korelacji
plt.figure(figsize=(14,8))
sns.heatmap(houses_copy[num_features].corr(), annot=True, cmap="RdBu_r", fmt=" .2f", linewidth=0.5)
plt.savefig('wykres_korelacji.png')
plt.show()

# histogram częstości
plt.figure(figsize=(8,6))
sns.histplot(houses_copy["Cena"], bins=40, kde=True, color="blue")
plt.title("Histogram ceny")
plt.xlabel("Cena")
plt.ylabel("Ilosc domow")
plt.savefig("histogram.png")
plt.show()

# wykres pudełkowy
plt.figure(figsize=(10,6))
sns.boxplot(x=houses_copy["Jakosc"], y=houses_copy["Cena"], hue=houses_copy["Jakosc"], palette="coolwarm", legend=False)
plt.title("Jakosc vs Cena")
plt.xlabel("Jakos")
plt.ylabel("Cena")
plt.savefig("wykres_pudelkowy.png")
plt.show()

# wizualizacja zależności między zmiennymi
plt.figure(figsize=(8,6))
sns.scatterplot(x=houses_copy["Powierzchnia"], y=houses_copy["Cena"], color="purple", alpha=0.6)
plt.title("Powierzchnia vs Cena")
plt.xlabel("Powierzchnia")
plt.ylabel("Cena")
plt.savefig("powierzchnia_vs_cena.png")
plt.show()

# wykres pudelkowy cena
plt.figure(figsize=(8,6))
sns.boxplot(x=houses_copy["Cena"], color='skyblue')
plt.title("Cena")
plt.xlabel("Cena")
plt.savefig("cena.png")
plt.show()

# wykres pudelkowy pokazujacy zaleznosc ceny od sasiedztwa
plt.figure(figsize=(12,6))
sns.boxplot(x=houses_copy['Sasiedztwo'], y=houses_copy["Cena"], hue=houses_copy["Sasiedztwo"], palette="coolwarm", legend=False)
plt.title("Sasiedztwo vs Cena")
plt.xlabel("Sasiedztwo")
plt.xticks(rotation=90)
plt.ylabel("Cena")
plt.savefig("sas_vs_cena.png")
plt.show()

# histogram ceny ze zbioru houses po jej logarytmowaniu (żeby zadzialal zakomentowac usuwanie wartosci odstajacych w houses.py)
houses_copy["Cena"] = np.log(houses_copy["Cena"])
plt.figure(figsize=(8,6))
sns.histplot(houses_copy["Cena"], bins=40, kde=True, color="blue")
plt.title("Histogram ceny po jej logarytmowaniu")
plt.xlabel("Logarytmowana cena")
plt.ylabel("Ilosc domow")
plt.savefig("histogram_ceny_log.png")
plt.show()
