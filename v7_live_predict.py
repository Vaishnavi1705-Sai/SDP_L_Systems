import serial
import time
import joblib
import numpy as np
from pythonosc import udp_client

# --- CONFIG ---
PORT = 'COM3'
BAUD = 115200
MODEL_PATH = 'biometric_model.pkl'
OSC_CLIENT = udp_client.SimpleUDPClient("127.0.0.1", 7000)

# 1. Load the Intelligence
print("Loading Biometric Brain...")
try:
    brain = joblib.load(MODEL_PATH)
    scaler = brain['scaler']
    pca = brain['pca']
    kmeans = brain['kmeans']
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2) # Wait for Arduino reset
    
    # --- STEP 2: DYNAMIC CALIBRATION ---
    print("\n" + "="*50)
    print("INITIAL CALIBRATION: DO NOT TOUCH SENSOR OR PLANT")
    print("="*50)
    
    baseline_samples = []
    for i in range(40):
        ser.write(b'G')
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if "," in line:
            try:
                c_val = float(line.split(",")[0])
                baseline_samples.append(c_val)
                print(f"\rSampling Ambient Air: {len(baseline_samples)}/40", end="")
            except: continue
        time.sleep(0.05)
    
    AIR_BASELINE = np.mean(baseline_samples)
    print(f"\n\nCalibration Complete.")
    print(f"Current Air Baseline: {AIR_BASELINE:.2f}")
    print("--- LIVE BIOMETRIC PREDICTION ENGINE ACTIVE ---")

    # --- STEP 3: MAIN PREDICTION LOOP ---
    while True:
        input("\n>>> Press ENTER to analyze a 1.25s touch...")
        
        raw_values = []
        temps = []
        ser.reset_input_buffer()

        while len(raw_values) < 50:
            ser.write(b'G')
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if "," in line:
                try:
                    c_str, t_str = line.split(",")
                    raw_values.append(int(float(c_str)))
                    temps.append(float(t_str))
                    print(f"\rReading DNA: {len(raw_values)}/50", end="")
                except: continue

        # Feature Extraction
        avg_raw = np.mean(raw_values)
        avg_temp = np.mean(temps)
        variance = np.var(raw_values)
        jaggedness = np.sum(np.abs(np.diff(raw_values)))
        slope = (raw_values[-1] - raw_values[0]) / 50
        sig_range = np.max(raw_values) - np.min(raw_values)
        
        # Calculate Delta (Difference from Air)
        delta_raw = avg_raw - AIR_BASELINE

        # Assemble features for AI
        new_features = np.array([[variance, slope, jaggedness, sig_range, avg_raw, avg_temp]])

        # Machine Learning Prediction
        scaled_feat = scaler.transform(new_features)
        pca_coords = pca.transform(scaled_feat)
        cluster_id = kmeans.predict(pca_coords)[0]

        # Send to TouchDesigner
        OSC_CLIENT.send_message("/dna/cluster", int(cluster_id))
        OSC_CLIENT.send_message("/dna/growth", float(avg_raw / 1024.0))
        
        # --- ENHANCED DEBUG DASHBOARD ---
        print(f"\n\n" + "-"*50)
        print(f"RESULT: Cluster {cluster_id}")
        print(f"PCA: X={pca_coords[0][0]:.2f}, Y={pca_coords[0][1]:.2f}")
        print(f"BIOMETRICS: Var={variance:.1f} | Delta from Air={delta_raw:+.1f}")
        print(f"STABILITY: Jaggedness={jaggedness:.1f} | Slope={slope:.2f}")
        print("-"*50)

except KeyboardInterrupt:
    print("\nClosing connection...")
finally:
    ser.close()