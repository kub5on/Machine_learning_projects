## Marketing Campaign Clustering 💻

This project performs **customer segmentation** based on marketing campaign data (`marketing_campaign.csv`).  
It uses **unsupervised learning methods** including **K-Means** and **Hierarchical Clustering**, combined with **dimensionality reduction** techniques like **PCA** and **UMAP**. Check it out and explore the clusters! ⚡

### Features
- **Data Preprocessing:** Label encoding for categorical features, scaling numerical features.  
- **Clustering Algorithms:**  
  - K-Means (on full dataset and on PCA/UMAP reduced data)  
  - Hierarchical Clustering (on UMAP-reduced data)  
- **Dimensionality Reduction:** PCA and UMAP for visualization and improved clustering performance.  
- **Evaluation Metrics:** Silhouette Score to assess cluster quality.  

## Dependencies
`Python 3.9+`, `pandas`, `numpy`, `scikit-learn`, `kneed`, `umap-learn`, `scipy`, `joblib`

## How to run
1. Add `marketing_campaign.csv` to the project folder.  
