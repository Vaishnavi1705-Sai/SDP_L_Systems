import joblib
import pandas as pd

# Load the brain
brain = joblib.load('biometric_model.pkl')
kmeans = brain['kmeans']
scaler = brain['scaler']
pca = brain['pca']

# The centers in 2D space
print("--- Cluster Centers (PCA Space) ---")
for i, center in enumerate(kmeans.cluster_centers_):
    print(f"Cluster {i}: X={center[0]:.2f}, Y={center[1]:.2f}")

# Inverse transform to see what these mean in REAL sensor units
# This shows you the "Thresholds" in terms of Variance, Raw Value, etc.
raw_centers = scaler.inverse_transform(pca.inverse_transform(kmeans.cluster_centers_))
centers_df = pd.DataFrame(raw_centers, columns=['Variance', 'Slope', 'Jaggedness', 'Range', 'AvgRaw', 'AvgTemp'])

print("\n--- What each Cluster 'Expects' (The Thresholds) ---")
print(centers_df)