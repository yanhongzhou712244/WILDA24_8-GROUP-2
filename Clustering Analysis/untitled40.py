# -*- coding: utf-8 -*-
"""Untitled40.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1r6zD_J7lnSjI0XF4u9TkLmbzX58X1VZN
"""

# Import necessary libraries
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the datasets
file_path_preprocessed = '/content/Cleaned_and_Perfect_Dataset.csv'
preprocessed_data = pd.read_csv(file_path_preprocessed)

# Select relevant features for clustering from the preprocessed dataset
X_preprocessed = preprocessed_data[['tenure', 'MonthlyCharges']]

# Scale the data
scaler = StandardScaler()
X_scaled_preprocessed = scaler.fit_transform(X_preprocessed)

# Determine the optimal number of clusters using the elbow method
inertia = []
for k in range(1, 6):  # Adjust the range as needed
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X_scaled_preprocessed)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve to visualize the optimal number of clusters
plt.plot(range(1, 6), inertia, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Assuming 3 is the optimal number of clusters based on the elbow plot
kmeans_preprocessed = KMeans(n_clusters=3, random_state=0)
preprocessed_data['Cluster'] = kmeans_preprocessed.fit_predict(X_scaled_preprocessed)

# Display the resulting clusters and cluster centers
print("Cluster centers:\n", kmeans_preprocessed.cluster_centers_)
print(preprocessed_data.head())