import matplotlib.pyplot as plt
from main import y_test_inv, pred_lstm, pred_gru
from autotuning import best_pred_lstm, best_pred_gru


"""wykres ROC 1"""
plt.figure(figsize=(8, 6))
plt.plot(y_test_inv, label='wartości rzeczywiste', color='black')
plt.plot(pred_lstm, label='predykacje lstm', color='yellow')
plt.plot(pred_gru, label='predykcje gru', color='red')
plt.title('Porównanie predykcji LSTM vs GRU')
plt.legend()
plt.savefig('LSTM_vs_GRU.png')
plt.show()


"""wykres ROC 2"""
plt.figure(figsize=(8, 6))
plt.plot(y_test_inv, label='wartości rzeczywiste', color='black')
plt.plot(best_pred_lstm, label='predykacje lstm', color='yellow')
plt.plot(best_pred_gru, label='predykcje gru', color='red')
plt.title('Porównanie predykcji LSTM vs GRU (po tuningu)')
plt.legend()
plt.savefig('LSTM_vs_GRU_tuned.png')
plt.show()
