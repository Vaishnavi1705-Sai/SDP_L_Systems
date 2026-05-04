import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# 1. Load Data and the "Brain"
df = pd.read_csv('set3_cleaned_new.csv', header=None).dropna()
brain = joblib.load('biometric_model.pkl')

scaler = brain['scaler']
pca = brain['pca']
kmeans = brain['kmeans']

# 2. Process Data through the AI Pipeline
X = df.iloc[:, -6:] # Extract features
X_scaled = scaler.transform(X) # Standardize
X_pca = pca.transform(X_scaled) # Reduce to 2D
clusters = kmeans.predict(X_pca) # Assign Cluster ID (0, 1, or 2)

# 3. Create the Visualization
plt.figure(figsize=(10, 7))
colors = ['#2ecc71', '#3498db', '#e74c3c'] # Green, Blue, Red
archetypes = ["Cluster 0: The Sequoia", "Cluster 1: The Willow", "Cluster 2: The Briar"]

for i in range(3):
    mask = (clusters == i)
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                c=colors[i], label=archetypes[i], s=70, edgecolors='white', alpha=0.8)

# Add the "Center" of each cluster (The AI's perfect example of that tree)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='yellow', s=200, marker='X', label='Cluster Center', edgecolors='black')

plt.title("The AI's Decision Map (Cluster Grouping)")
plt.xlabel("Intensity (How hard you touch)")
plt.ylabel("Stability (How much you tremble)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()