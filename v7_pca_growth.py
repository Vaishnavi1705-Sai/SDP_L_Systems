import serial
import time
import numpy as np
from pythonosc import udp_client
import joblib

# --- CONFIG ---
PORT = 'COM3'
BAUD = 115200
OSC_CLIENT = udp_client.SimpleUDPClient("127.0.0.1", 7000)

print("Starting Biometric Brain (PCA-Driven Growth Mode)...")

try:
    # LOAD TRAINED ML MODEL
    print("Loading ML Model (PCA + Scaler)...")
    model_data = joblib.load('biometric_model.pkl')
    scaler = model_data['scaler']
    pca = model_data['pca']
    kmeans = model_data['kmeans']
    print("✓ ML Model Loaded Successfully\n")
    
    ser = serial.Serial(PORT, BAUD, timeout=0.1)
    time.sleep(2)
    
    # --- DYNAMIC CALIBRATION ---
    print("CALIBRATING... DO NOT TOUCH PLANT")
    baseline_samples = []
    while len(baseline_samples) < 40:
        ser.write(b'G')
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if "," in line:
            try:
                c_val = float(line.split(",")[0])
                baseline_samples.append(c_val)
            except: continue
    
    AIR_BASELINE = np.mean(baseline_samples)
    print(f"Air Baseline: {AIR_BASELINE:.2f} | SYSTEM LIVE\n")

    # --- CONTINUOUS STREAMING ---
    while True:
        raw_values = []
        temp_values = []
        
        # Capture 20 samples per frame for instant calculation
        while len(raw_values) < 20:
            ser.write(b'G')
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if "," in line:
                try:
                    parts = line.split(",")
                    if len(parts) == 2:
                        c_str, t_str = parts
                        raw_values.append(int(float(c_str)))
                        temp_values.append(float(t_str))
                except: continue

        # 1. CALCULATE ALL 6 FEATURES
        avg_raw = np.mean(raw_values)
        variance = np.var(raw_values)
        delta_raw = avg_raw - AIR_BASELINE
        
        # Additional features (to match training)
        slope = (raw_values[-1] - raw_values[0]) / len(raw_values) if len(raw_values) > 1 else 0
        jaggedness = sum(abs(raw_values[j] - raw_values[j-1]) for j in range(1, len(raw_values)))
        sig_range = max(raw_values) - min(raw_values) if raw_values else 0
        avg_temp = np.mean(temp_values)

        # 2. TRANSFORM THROUGH PCA
        features = np.array([[variance, slope, jaggedness, sig_range, avg_raw, avg_temp]])
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)
        
        # Extract PCA components
        pc1 = features_pca[0, 0]  # PC1 = Intensity (how hard you're touching)
        pc2 = features_pca[0, 1]  # PC2 = Stability (how clean the signal is)
        
        # 3. GROWTH LOGIC - BACK TO THRESHOLD-BASED DELTA_RAW
        # This was working well, so keep it
        if delta_raw < 35:
            growth_val = 0.0
        else:
            growth_val = min((delta_raw / 160.0), 1.0)

        # 4. MOISTURE DETECTION - NOW USING PCA FEATURES
        # Get cluster prediction from K-means
        cluster = kmeans.predict(features_pca)[0]
        
        # Map cluster to moisture state
        # Cluster 0 = DRY (high noise)
        # Cluster 1 = NORMAL (balanced)
        # Cluster 2 = WET (flat signal)
        
        cluster_names = ["DRY   ", "NORMAL", "WET   "]
        state_name = cluster_names[cluster]
        
        # Convert cluster to smooth [0.0, 1.0] using probabilities
        distances = kmeans.transform(features_pca)[0]
        cluster_prob = np.exp(-distances) / np.sum(np.exp(-distances))  # Softmax
        
        # Smooth moisture mapping: Cluster 0→0.0, Cluster 1→0.5, Cluster 2→1.0
        moisture_state = (
            cluster_prob[0] * 0.0 +  # DRY probability
            cluster_prob[1] * 0.5 +  # NORMAL probability
            cluster_prob[2] * 1.0    # WET probability
        )

        # 5. SEND TO TOUCHDESIGNER
        OSC_CLIENT.send_message("/dna/cluster", float(moisture_state)) 
        OSC_CLIENT.send_message("/dna/growth", float(growth_val))
        
        # Debug output
        print(f"\rPCA-MOISTURE: [{state_name}] Moisture(PCA): {moisture_state:.2f} | Growth(Delta): {growth_val:.2f} "
              f"| Delta: {delta_raw:+.1f} | Cluster: {cluster} | PC2: {pc2:+.2f}", end="")
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\n\nClosing...")
except FileNotFoundError:
    print("ERROR: Could not find 'biometric_model.pkl'. Please run train_model.py first.")
finally:
    if 'ser' in locals():
        ser.close()
        print("Serial connection closed.")
