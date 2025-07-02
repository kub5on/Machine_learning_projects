import os
os.environ['LOKY_MAX_CPU_COUNT'] = '0'

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from umap import UMAP
import joblib
import zipfile

# Ustawienie ziarna losowego dla powtarzalności wyników
np.random.seed(42)

df = pd.read_csv('marketing_campaign.csv')
missing_values = df.isnull().sum()

"""enkodowanie danych kategorycznych"""
encoder = LabelEncoder()
df['Wyksztalcenie'] = encoder.fit_transform(df['Wyksztalcenie'])
df['MieszkaZ'] = encoder.fit_transform(df['MieszkaZ'])

# df_categoricals = df.select_dtypes(include=["object"])  # to samo co wyżej ale inaczej zapisane
# df_encoded_categoricals = df_categoricals.apply(label_encoder.fit_transform)
#
# df[df_encoded_categoricals.columns] = df_encoded_categoricals

"""skalowanie danych"""
df_numerical = df.select_dtypes(include=[np.number])
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_numerical), columns=df_numerical.columns)
df[df_numerical.columns] = df_scaled

"""klastrowanie danych"""
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", max_iter=300, n_init=10, random_state=42)
    kmeans.fit(df)
    wcss.append(kmeans.inertia_)


kl = KneeLocator(range(1, 11), wcss, curve="convex", direction="decreasing")
optimal_clusters = kl.elbow


"""redukcja wymiarów PCA"""
pca = PCA(n_components=2)
df_pca = pd.DataFrame(pca.fit_transform(df), columns=["PC1", "PC2"])


"""silhouette score"""
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df_pca["KMeans_Cluster"] = kmeans.fit_predict(df_pca)

kmeans_score = silhouette_score(df_pca, kmeans.labels_)


"""redukcja wymiarów UMAP"""
umap_model = UMAP(n_components=2, random_state=42)
df_umap = pd.DataFrame(umap_model.fit_transform(df), columns=["UMAP1", "UMAP2"])

joblib.dump(umap_model, 'umap_model.pkl') # zapisanie plików
df_umap.to_csv("df_umap.csv", index=False)

with zipfile.ZipFile('umap.zip', 'w') as zipf: # utworzenie pliku zip
    zipf.write('umap_model.pkl')
    zipf.write('df_umap.csv')
