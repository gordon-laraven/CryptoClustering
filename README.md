# CryptoClustering README

## Overview

This repository documents the `CryptoClustering` project, where we apply K-means clustering and Principal Component Analysis (PCA) to classify cryptocurrencies based on their price fluctuations over various timeframes. Specifically, we analyze price changes across intervals spanning 24 hours, 7 days, 30 days, 60 days, 200 days, and 1 year.

## Table of Contents
- [Sources](#sources)
- [Project Structure](#project-structure)
- [Instructions](#instructions)
- [Data Preparation](#data-preparation)
- [Clustering](#clustering)
- [Principal Component Analysis](#principal-component-analysis)
- [Visualization](#visualization)
- [Conclusion](#conclusion)

## Sources
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

## Project Structure
```
CryptoClustering/
├── Crypto_Clustering.ipynb
├── Resources/
│   └── crypto_market_data.csv
└── README.md
```

## Instructions

1. **Create a New Repository**: Create a new GitHub repository named `CryptoClustering` and clone it to your local machine.

2. **Download Required Files**: Download the [Module 11 Challenge files](https://static.bc-edx.com/ai/ail-v-1-0/m11/lms/starter/M11_Starter_Code.zip) and extract them in your repository folder.

3. **Prepare the Notebook**: Rename `Crypto_Clustering_starter_code.ipynb` to `Crypto_Clustering.ipynb`.

4. **Load Data**: Load the `crypto_market_data.csv` into a DataFrame and set the index to the `coin_id` column.

5. **Summary Statistics**: Use the `describe()` method to get summary statistics for the dataset.

## Data Preparation

1. **Normalization**: Use the `StandardScaler` from `scikit-learn` to normalize the market data.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the data
market_data_df = pd.read_csv("Resources/crypto_market_data.csv", index_col="coin_id")

# Normalize data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(market_data_df)
```

2. **Create Scaled DataFrame**: Store the scaled results in a new DataFrame while preserving the `coin_id`.

```python
scaled_df = pd.DataFrame(scaled_data, index=market_data_df.index, columns=market_data_df.columns)
```

## Clustering

1. **Find the Best k Value**: Implement the elbow method to determine the optimal number of clusters (`k`) for K-means.

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

k_values = list(range(1, 12))
inertia = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(scaled_df)
    inertia.append(kmeans.inertia_)

plt.plot(k_values, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.grid()
plt.show()

# Determine the best k visually (typically where the curve flattens)
```

2. **Cluster Data**: Use the best `k` value to fit the K-means model and predict clusters.

```python
best_k = 4  # Replace this with your determined best k value
kmeans = KMeans(n_clusters=best_k, random_state=0)
kmeans.fit(scaled_df)
clusters = kmeans.predict(scaled_df)
```

3. **Scatter Plot**: Visualize the clustering result.

```python
scaled_df['cluster'] = clusters
scaled_df.plot.scatter(x='price_change_percentage_24h', y='price_change_percentage_7d', c='cluster', colormap='rainbow')
plt.title('Cryptocurrency Clusters')
plt.show()
```

## Principal Component Analysis

1. **PCA Implementation**: Reduce dimensions to three principal components.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca_result = pca.fit_transform(scaled_df)
```

2. **Explained Variance**: Analyze the variance explained by each principal component.

```python
explained_variance = pca.explained_variance_ratio_
print(f'Total explained variance: {explained_variance.sum()}')
```

## Visualization

- **Elbow Method for PCA Data**: Repeat the elbow method for the PCA-transformed data.

- **Scatter Plot with PCA**: Display clusters on a scatter plot using PCA components.

## Conclusion

- Reflect on whether using PCA affected the clustering and the implications of the derived features based on their weights in PCA.
