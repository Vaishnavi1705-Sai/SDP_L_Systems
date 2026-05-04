import serial
import time
import numpy as np
from pythonosc import udp_client
import joblib

# --- CONFIG ---
PORT = 'COM3'
BAUD = 115200
OSC_CLIENT = udp_client.SimpleUDPClient("127.0.0.1", 7000)

print("Starting Biometric Brain (ML-Integrated Mode)...")

try:
    # LOAD TRAINED ML MODEL
    print("Loading ML Intelligence Layer...")
    model_data = joblib.load('biometric_model.pkl')
    scaler = model_data['scaler']
    pca = model_data['pca']
    kmeans = model_data['kmeans']
    print("✓ ML Model Loaded Successfully")
    
    ser = serial.Serial(PORT, BAUD, timeout=0.1)
    time.sleep(2)
    
    # --- DYNAMIC CALIBRATION ---
    print("\nCALIBRATING... DO NOT TOUCH PLANT")
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
    print(f"Air Baseline: {AIR_BASELINE:.2f} | SYSTEM LIVE")

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

        # 1. CALCULATE CORE FEATURES
        avg_raw = np.mean(raw_values)
        variance = np.var(raw_values)
        delta_raw = avg_raw - AIR_BASELINE
        
        # Calculate additional features (to match training features)
        slope = (raw_values[-1] - raw_values[0]) / len(raw_values) if len(raw_values) > 1 else 0
        jaggedness = sum(abs(raw_values[j] - raw_values[j-1]) for j in range(1, len(raw_values)))
        sig_range = max(raw_values) - min(raw_values) if raw_values else 0
        avg_temp = np.mean(temp_values)

        # 2. GROWTH LOGIC (0.0 to 1.0 based strictly on pressure)
        if delta_raw < 35:
            growth_val = 0.0
        else:
            # Map pressure to a smooth 0.0 -> 1.0 curve
            growth_val = min((delta_raw / 160.0), 1.0)

        # 3. ML-BASED MOISTURE STATE CLASSIFICATION
        # Prepare features for ML model (same order as training)
        features = np.array([[variance, slope, jaggedness, sig_range, avg_raw, avg_temp]])
        
        # Transform through PCA pipeline
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)
        
        # Get cluster prediction
        cluster = kmeans.predict(features_pca)[0]
        
        # Map clusters to moisture states
        # Cluster 0 = Low variance/stable = WET (1.0)
        # Cluster 1 = Mid stability = NORMAL (0.5)
        # Cluster 2 = High noise/jitter = DRY (0.0)
        
        # Optional: Use cluster probability for smoother transitions
        distances = kmeans.transform(features_pca)[0]  # Distance to each cluster
        cluster_prob = np.exp(-distances) / np.sum(np.exp(-distances))  # Softmax
        
        # Create smooth moisture state from probabilities
        moisture_state = (
            cluster_prob[0] * 1.0 +      # WET probability
            cluster_prob[1] * 0.5 +      # NORMAL probability  
            cluster_prob[2] * 0.0        # DRY probability
        )
        
        # State name for debugging
        state_names = ["WET", "NORMAL", "DRY"]
        state_name = state_names[cluster]

        # 4. SEND TO TOUCHDESIGNER
        OSC_CLIENT.send_message("/dna/cluster", float(moisture_state)) 
        OSC_CLIENT.send_message("/dna/growth", float(growth_val))
        
        print(f"\rML: [{state_name}] Moisture: {moisture_state:.2f} | Growth: {growth_val:.2f} | "
              f"Delta: {delta_raw:+.1f} | Var: {variance:.1f}", end="")
        time.sleep(0.05) # Keep loop stable

except KeyboardInterrupt:
    print("\nClosing...")
except FileNotFoundError:
    print("ERROR: Could not find 'biometric_model.pkl'. Please train the model first using train_model.py")
finally:
    if 'ser' in locals():
        ser.close()
