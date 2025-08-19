from main import wcss, df_pca, df_umap, df
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


"""wykres przedstawiający podział na klastry po PCA"""
sns.scatterplot(data=df_umap, x="UMAP1", y="UMAP2", hue="Hierarchical_Cluster", palette="magma", s=100, edgecolor="black")
plt.title("Hierarchical Clustering po UMAP")
plt.savefig('hier_clustering_umap')
plt.show()


# przynależność różnych cech do klastrów
features = ['WydatkiOwoce', 'WydatkiRyby', 'WydatkiSlodycze']
df_combined = df.copy()
df_combined["Cluster"] = df_umap['KMeans_Cluster']

fig, axes = plt.subplots(1, 3, figsize=(18,5))
for (i, feature) in enumerate(features):
    sns.boxplot(x='Cluster', y=feature, data=df_combined, ax=axes[i], color='green')
    axes[i].set_title(f'{feature}')
    axes[i].set_xlabel('Klaster')
    axes[i].set_ylabel('Wartość cechy')

plt.suptitle('Wykres przedstawiający przynależność cech do klastrów po KMeans na UMAP')
plt.tight_layout()
plt.savefig("features_plot_kmeans_umap")
plt.show()

