from main import wcss, df_pca
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

"""wizualizacja wcss (elbow method)"""
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow method')
plt.xticks(np.arange(1, 11, 1))
plt.xlabel('Cluster quantity')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

"""wykres danych po redukcji wymiarów (n_components=2)"""
plt.figure(figsize=(7,6))
plt.scatter(df_pca['PC1'], df_pca["PC2"], alpha=0.5, c="red")
plt.title("PCA: redukcja do dwóch wymiarów")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

"""wykres przedstawiający podział na klastry po PCA"""
sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="KMeans_Cluster", palette="plasma", s=100, edgecolor="black")
plt.title("KMeans Clustering po PCA")
plt.savefig('pca_clustering')
plt.show()

