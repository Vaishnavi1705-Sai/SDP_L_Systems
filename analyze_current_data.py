import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load Data
try:
    df = pd.read_csv('biometric_data.csv', header=None)
    df = df.dropna()
    print(f"--- DATASET PROFILE ---")
    print(f"Total valid trials: {len(df)}")
except:
    print("CSV not found!")
    exit()

# Extract only the last 6 columns for math
features = df.iloc[:, -6:] 
features.columns = ['Variance', 'Slope', 'Jaggedness', 'Range', 'AvgRaw', 'AvgTemp']
labels = df.iloc[:, 0]

# --- TEST 1: STABILITY ANALYSIS ---
print("\n--- TEST 1: STABILITY ANALYSIS (Mean per Person) ---")
# Fixed GroupBy Logic
analysis = df.groupby(0).mean().iloc[:, -6:]
analysis.columns = ['Variance', 'Slope', 'Jaggedness', 'Range', 'AvgRaw', 'AvgTemp']
print(analysis[['Variance', 'Jaggedness', 'AvgRaw']])

# --- TEST 2: PCA INFLUENCE ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\n--- TEST 2: PCA COMPONENT WEIGHTS ---")
components = pd.DataFrame(pca.components_, columns=features.columns, index=['X-Axis', 'Y-Axis'])
print(components)

# --- TEST 3: THE MAP ---
print("\n--- TEST 3: GENERATING SCATTERPLOT... ---")
plt.figure(figsize=(12, 7))

# Plot each person with a different color
for label in labels.unique():
    mask = labels == label
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=label, s=60, alpha=0.7)

plt.title("Biometric Map: How the AI 'Sees' Your Data")
plt.xlabel("Principal Component 1 (Intensity)")
plt.ylabel("Principal Component 2 (Stability)")
plt.axhline(0, color='black', linewidth=0.5, ls='--')
plt.axvline(0, color='black', linewidth=0.5, ls='--')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.grid(alpha=0.3)
plt.tight_layout()

print("Plot is opening in a new window. Check your taskbar!")
plt.show()