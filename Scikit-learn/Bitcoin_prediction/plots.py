import pandas as pd
import matplotlib.pyplot as plt
from main import bitcoin, results, predictions


def plot_predictions(model_name, predictions, y):
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(y.index, y, label='Wartości rzeczywiste', linestyle='dashed', color='purple')

    for i, (idx, y_pred) in enumerate(predictions):
        ax.plot(idx, y_pred, label=f"{model_name}: próbka {i + 1}")

    ax.set_title(f"Wykres dla modelu: {model_name}")
    ax.set_xlabel("Data")
    ax.set_ylabel("Cena bitcoina")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{model_name}.png')
    plt.show()




"""wykres cen bitcoina w czasie"""
# plt.figure(figsize=(12,6))
# plt.plot(bitcoin.index, bitcoin['value'], label='Wartość bitcoina', color='yellow')
# plt.xlabel("Czas")
# plt.ylabel("Cena")
# plt.title("Wykres zmian ceny bitcoina w czasie")
# plt.legend()
# plt.grid(True)
# plt.savefig('ceny_bitcoina')
# plt.show()


"""wizualizacja wyników - porównanie modeli z rzeczywistymi cenami bitcoina"""
for name, preds in predictions.items():
    plot_predictions(name, preds, bitcoin['value'])
