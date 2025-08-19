import shap
import matplotlib.pyplot as plt
import pandas as pd
from main import shap_values, X_test, explainer, y_test
import matplotlib

# wybieramy tylko osoby z chorobą serca
X_test_1 = X_test[y_test == 1]
shap_values_1 = shap_values[(y_test == 1).values]


shap.summary_plot(shap_values, X_test, show=False)
# plt.savefig("shap_summary_target1.png", bbox_inches='tight')
plt.show()


with open("interpretacja_shap.txt", "w") as f:
    f.write(
                'Wykres przedstawia wpływ poszczególnych cech na wynik predykcji modelu.\n'
                'Każda kropka to jedna obserwacja; oś X to wartość SHAP, czyli wpływ cechy na wynik,\n' 
                'a kolor oznacza wartość cechy (od niskich – niebieski, do wysokich – czerwony).\n\n'
                
                'Największy wpływ na wynik osób, u których wykryto choroby serca (target=1) mają cechy: cp, oldpeak, exang, ca.\n'
                '- Wysokie wartości cp zwiększają prawdopodobieństwo pozytywnej klasy.\n'
                '- Wysokie wartości oldpeak i exang zmniejszają wynik.\n'
                '- Wysokie wartości ca i thal również zmniejszają wynik, podczas gdy slope działa odwrotnie.\n\n'
                
                'Pozostałe cechy (thalach, fbs, restecg, chol, trestbps, sex, age) mają mniejszy wpływ.\n'
    )


