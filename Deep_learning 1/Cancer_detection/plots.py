import matplotlib.pyplot as plt
import seaborn as sns
from main import cm, label_encoder, history, model, X_train, X_test, X
import shap

"""confusion matrix"""
# plt.figure(figsize=(6, 5))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
#             xticklabels=label_encoder.classes_,
#             yticklabels=label_encoder.classes_)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.savefig('confusion_matrix.png')
# plt.show()

"""training history plot"""
# plt.plot(history.history['accuracy'], label="Train accuracy", color='green')
# plt.plot(history.history['val_accuracy'], label="Val accuracy", color='orange')
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.title("Model training accuracy")
# plt.savefig('training_plot')
# plt.show()

"""SHAP summary_plot"""
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test[:100])
shap.summary_plot(shap_values, features=X_test[:100], feature_names=X.columns, show=False)
plt.savefig("shap_summary_plot.png")

