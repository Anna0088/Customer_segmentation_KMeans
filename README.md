# Customer_segmentation_KMeans
# Customer Segmentation using K-Means Clustering

This project demonstrates how to perform customer segmentation using the K-Means clustering algorithm on the "Mall Customers" dataset. The goal is to group customers into distinct clusters based on their annual income and spending score, which can help businesses tailor marketing strategies and improve customer satisfaction.

## Table of Contents

- [Introduction](#introduction)
- [Dataset Description](#dataset-description)
- [Prerequisites](#prerequisites)
- [Code Explanation](#code-explanation)
  - [Importing Libraries](#importing-libraries)
  - [Loading the Dataset](#loading-the-dataset)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Selecting Features](#selecting-features)
  - [Elbow Method to Determine Optimal Clusters](#elbow-method-to-determine-optimal-clusters)
  - [Applying K-Means Clustering](#applying-k-means-clustering)
  - [Visualizing the Clusters](#visualizing-the-clusters)
- [Results and Interpretation](#results-and-interpretation)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction

Customer segmentation is a vital process in marketing, allowing businesses to target specific groups of customers effectively. By clustering customers based on their purchasing behavior and demographics, companies can design personalized marketing campaigns, improve customer service, and increase profitability.

## Dataset Description

The "Mall Customers" dataset contains information about customers, including:

- **CustomerID**: Unique identifier for each customer.
- **Gender**: Gender of the customer.
- **Age**: Age of the customer.
- **Annual Income (k$)**: Annual income of the customer in thousand dollars.
- **Spending Score (1-100)**: A score assigned by the mall based on customer behavior and spending nature.

The dataset can be downloaded from [Kaggle](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python).

## Prerequisites

- Python 3.x
- Libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn

Install the required libraries using:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Code Explanation

### Importing Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
```

- **NumPy**: For numerical computations.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib & Seaborn**: For data visualization.
- **Scikit-learn's KMeans**: For implementing the K-Means clustering algorithm.

### Loading the Dataset

```python
# Loading the data from CSV file into a Pandas DataFrame
customer_data = pd.read_csv('/content/Mall_Customers.csv')

# Displaying the first 5 rows of the DataFrame
customer_data.head()
```

- The dataset is loaded into a DataFrame called `customer_data`.
- `customer_data.head()` displays the first five rows to give an overview of the data.

### Exploratory Data Analysis (EDA)

```python
# Finding the number of rows and columns
customer_data.shape

# Getting information about the dataset
customer_data.info()

# Checking for missing values
customer_data.isnull().sum()
```

- **Shape**: Provides the dimensions of the DataFrame.
- **Info**: Displays the data types and non-null counts.
- **Missing Values**: Checks if there are any null values in the dataset.

### Selecting Features

```python
# Selecting the Annual Income and Spending Score columns
X = customer_data.iloc[:, [3, 4]].values

print(X)
```

- **Feature Selection**: We select **Annual Income** and **Spending Score** as features for clustering.
- `iloc[:, [3, 4]]` selects the 4th and 5th columns (indexing starts from 0).
- `X` is a NumPy array containing the selected features.

### Elbow Method to Determine Optimal Clusters

```python
# Finding WCSS value for different number of clusters
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Graph
sns.set()
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
```

- **Within-Cluster Sum of Squares (WCSS)**: Measures the sum of squares of distances of samples to their closest cluster center.
- **Elbow Method**: Plots WCSS against the number of clusters to find the "elbow point" where the rate of decrease sharply changes, indicating the optimal number of clusters.
- **Loop**: Iterates `k` from 1 to 10, computes WCSS, and appends it to the `wcss` list.
- **Plot**: Visualizes the WCSS values to identify the elbow point.

### Applying K-Means Clustering

```python
# Applying KMeans with the optimal number of clusters found (k=5)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)

# Predicting the cluster labels
Y = kmeans.fit_predict(X)

print(Y)
```

- **KMeans Initialization**: We set `n_clusters=5` based on the elbow method.
- **Fit Predict**: Computes cluster centers and predicts the cluster for each sample.
- **Cluster Labels**: Stored in array `Y`, indicating which cluster each data point belongs to.

### Visualizing the Clusters

```python
# Plotting the clusters and their centroids
plt.figure(figsize=(8, 8))

# Cluster 1
plt.scatter(X[Y == 0, 0], X[Y == 0, 1], s=50, c='green', label='Cluster 1')

# Cluster 2
plt.scatter(X[Y == 1, 0], X[Y == 1, 1], s=50, c='red', label='Cluster 2')

# Cluster 3
plt.scatter(X[Y == 2, 0], X[Y == 2, 1], s=50, c='yellow', label='Cluster 3')

# Cluster 4
plt.scatter(X[Y == 3, 0], X[Y == 3, 1], s=50, c='violet', label='Cluster 4')

# Cluster 5
plt.scatter(X[Y == 4, 0], X[Y == 4, 1], s=50, c='blue', label='Cluster 5')

# Plotting the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=100, c='cyan', label='Centroids')

plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
```

- **Figure Size**: Sets the size of the plot.
- **Scatter Plots**: Plots data points for each cluster with different colors.
- **Centroids**: Plots the cluster centers.
- **Labels and Title**: Adds titles and labels for clarity.
- **Legend**: Displays the legend to identify clusters.

**Note**: In the original code, there was a typo in the plotting function (`X[Y=3, 0]` should be `X[Y == 3, 0]`). This has been corrected.

## Results and Interpretation

The clustering resulted in five distinct customer segments:

1. **Cluster 1 (Green)**:
   
   - **Low Income**, **Low Spending Score**
   - Customers with low annual income and low spending habits.

2. **Cluster 2 (Red)**:
   
   - **Low Income**, **High Spending Score**
   - Customers with low annual income but high spending habits.

3. **Cluster 3 (Yellow)**:
   
   - **High Income**, **High Spending Score**
   - Affluent customers who spend a lot; potentially the most profitable segment.

4. **Cluster 4 (Violet)**:
   
   - **High Income**, **Low Spending Score**
   - Wealthy customers who spend less; potential to increase spending through targeted marketing.

5. **Cluster 5 (Blue)**:
   
   - **Average Income**, **Average Spending Score**
   - Average customers; stable segment.

The centroids represent the mean values of the clusters and help in understanding the central tendency of each group.

## Conclusion

By applying K-Means clustering to the mall customers dataset, we've successfully segmented the customers into distinct groups based on their annual income and spending score. This segmentation allows businesses to:

- Tailor marketing strategies to each customer group.
- Identify high-value customers.
- Develop targeted promotions to increase spending in lower-spending segments.
- Improve overall customer satisfaction by understanding customer needs.



