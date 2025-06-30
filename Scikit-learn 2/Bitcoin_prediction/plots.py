import pandas as pd
import matplotlib.pyplot as plt
from main import bitcoin

plt.figure(figsize=(12,6))
plt.plot(bitcoin.index, bitcoin['value'], label='Wartość bitcoina', color='yellow')
plt.xlabel("Czas")
plt.ylabel("Cena")
plt.title("Wykres zmian ceny bitcoina w czasie")
plt.legend()
plt.grid(True)
plt.savefig('ceny_bitcoina')
# plt.show()
