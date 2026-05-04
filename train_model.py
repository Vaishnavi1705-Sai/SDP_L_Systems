import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib

# 1. Load your data
try:
    df = pd.read_csv('set3_cleaned_new.csv', header=None)
    print(f"Loaded {len(df)} total rows.")
except Exception as e:
    print(f"Error: Could not find CSV file. {e}")
    exit()

# 2. CLEANING STEP: Remove any rows with missing data (NaN)
# This fixes the "ValueError: Input X contains NaN"
df = df.dropna()
print(f"Rows remaining after removing empty values: {len(df)}")

# 3. Extract only the Feature Columns (the last 6 columns)
X = df.iloc[:, -6:] 

# 4. Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Apply PCA (Reduce to 2 Dimensions)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 6. Create 3 Clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_pca)

# 7. SAVE EVERYTHING
model_data = {
    'scaler': scaler,
    'pca': pca,
    'kmeans': kmeans
}
joblib.dump(model_data, 'biometric_model.pkl')

print("\n--- SUCCESS ---")
print("Intelligence Layer Trained and Saved as 'biometric_model.pkl'!")
print(f"Final Cluster Count: {len(np.unique(kmeans.labels_))}")