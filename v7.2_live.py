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
    print(f"CRITICAL ERROR: Could not load model. Error: {e}")
    exit()

try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2) # Reset time
    
    # --- STEP 2: DYNAMIC AIR CALIBRATION ---
    # This solves the 'Environmental Drift' issue
    print("\n" + "="*50)
    print("CALIBRATING BASELINE: DO NOT TOUCH PLANT")
    print("="*50)
    
    baseline_samples = []
    while len(baseline_samples) < 40:
        ser.write(b'G')
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if "," in line:
            try:
                c_val = float(line.split(",")[0])
                baseline_samples.append(c_val)
                print(f"\rSniffing DNA Atmosphere: {len(baseline_samples)}/40", end="")
            except: continue
        time.sleep(0.01)
    
    AIR_BASELINE = np.mean(baseline_samples)
    print(f"\n\nAir Baseline set at: {AIR_BASELINE:.2f}")
    print("ENGINE READY. SYSTEM LIVE.")

    # --- STEP 3: PREDICTION LOOP ---
    while True:
        input("\n>>> [READY] Press ENTER to capture DNA...")
        
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
                    print(f"\rSequencing: {len(raw_values)}/50", end="")
                except: continue

        # Feature Extraction
        avg_raw = np.mean(raw_values)
        avg_temp = np.mean(temps)
        variance = np.var(raw_values)
        jaggedness = np.sum(np.abs(np.diff(raw_values)))
        slope = (raw_values[-1] - raw_values[0]) / 50
        sig_range = np.max(raw_values) - np.min(raw_values)
        
        # Calculate Delta (Relative change from the room baseline)
        delta_raw = avg_raw - AIR_BASELINE

        # Assemble features for AI
        new_features = np.array([[variance, slope, jaggedness, sig_range, avg_raw, avg_temp]])

        # Step 4: ML Pipeline
        scaled_feat = scaler.transform(new_features)
        pca_coords = pca.transform(scaled_feat)
        cluster_id = kmeans.predict(pca_coords)[0]

        # Step 5: HYBRID LOGIC OVERRIDES (The Safety Net)
        # Trees: 0 = Sequoia, 1 = Willow, 2 = Briar
        
        if delta_raw < 35:
            # Case: Too close to baseline to be a real touch
            final_tree = 0  # Sequoia
            behavior = "IDLE (Sequoia)"
        elif 35 <= delta_raw < 160:
            # Case: Clear signal change, but not maxed out
            final_tree = 1  # Willow
            behavior = "LIGHT TOUCH (Willow)"
        else:
            # Case: Delta > 160 or clipping (Var near 0 at high raw)
            final_tree = 2  # Briar
            behavior = "INTENSE TOUCH (Briar)"

        # 6. Communicate with TouchDesigner
        OSC_CLIENT.send_message("/dna/cluster", int(final_tree))
        OSC_CLIENT.send_message("/dna/growth", float(avg_raw / 1024.0))
        
        # --- THE DEBUG DASHBOARD ---
        print(f"\n\n" + "-"*50)
        print(f"AI PREDICTION : Cluster {cluster_id}")
        print(f"HYBRID RESULT : {behavior}")
        print(f"X/Y COORDS    : X={pca_coords[0][0]:.2f}, Y={pca_coords[0][1]:.2f}")
        print(f"BIOMETRICS    : Var={variance:.1f} | Delta={delta_raw:+.1f} | Raw={avg_raw:.1f}")
        print("-" * 50)

except KeyboardInterrupt:
    print("\nPowering down brain...")
finally:
    ser.close()