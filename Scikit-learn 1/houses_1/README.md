
# Analiza i predykcja cen domów

Ten projekt pokazuje kompletny pipeline analizy danych i budowania modeli regresyjnych do przewidywania cen domów na podstawie danych z pliku `houses.csv`.
Wykorzystuje popularne biblioteki Pythona: **pandas, scikit-learn, xgboost, matplotlib, seaborn**.

---

## Zawartość projektu

- Wczytanie danych z CSV
- Eksploracyjna analiza danych (EDA) poprzez wizualizacje:
  - Macierz korelacji
  - Histogram ceny
  - Wykres pudełkowy (jakość vs cena)
  - Wykres punktowy (powierzchnia vs cena)
  - Wykres pudełkowy dla ceny
  - Wykres pudełkowy (sąsiedztwo vs cena)
  - Histogram logarytmowanej ceny
- Uzupełnianie braków (`SimpleImputer`)
- Usuwanie wartości odstających (1.5 IQR)
- Kodowanie zmiennych kategorycznych (`OneHotEncoder`)
- Standaryzacja zmiennych numerycznych (`StandardScaler`)
- Podział danych na treningowe/testowe (80/20)
- Trenowanie i porównanie modeli:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - XGBoost
- Ewaluacja modeli przy użyciu:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - R² (R-squared)
- Wizualizacje:
  - Wykres ważności cech (XGBoost)
  - Wykresy rzeczywiste vs przewidywane

---

## Uruchomienie

### Wymagania

- Python ≥ 3.8
- Zależności z pliku [`requirements.txt`](./requirements.txt)

### Instalacja zależności

```bash
pip install -r requirements.txt
```

### Uruchomienie skryptu

1. Upewnij się, że plik `houses.csv` znajduje się w tym samym folderze co skrypt.  
2. Uruchom skrypt główny (przygotowanie danych i modele) oraz skrypt do wizualizacji:

```bash
python nazwa_skryptu.py
python wizualizacje.py
```

3. Po uruchomieniu skrypty wygenerują:
- `predicted_vs_real.png` — wykresy rzeczywiste vs przewidywane
- `Wykres_cech_xgboost.png` — wykres ważności cech
- `wykres_korelacji.png` — macierz korelacji
- `histogram.png` — histogram ceny
- `wykres_pudelkowy.png` — wykres pudełkowy (jakość vs cena)
- `powierzchnia_vs_cena.png` — wykres punktowy (powierzchnia vs cena)
- `cena.png` — wykres pudełkowy ceny
- `sas_vs_cena.png` — wykres pudełkowy (sąsiedztwo vs cena)
- `histogram_ceny_log.png` — histogram logarytmowanej ceny

Wyniki metryk modeli wyświetlą się również w konsoli w tabeli `results_df`.

---

## Struktura kodu

- Wczytywanie i kopiowanie danych
- Eksploracyjna analiza danych — wykresy
- Imputacja braków
- Usuwanie wartości odstających
- Kodowanie zmiennych kategorycznych
- Podział na `train/test`
- Standaryzacja
- Trenowanie i predykcja modeli
- Obliczanie metryk
- Wizualizacje wyników

---

## Technologie

- Python
- pandas
- numpy
- scikit-learn
- xgboost
- seaborn
- matplotlib

---

## Kontakt

W razie pytań lub sugestii — zapraszam do kontaktu.
