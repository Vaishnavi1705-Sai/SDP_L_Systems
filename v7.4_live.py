# live prediction without pressing enter
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

print("Loading Biometric Brain...")
brain = joblib.load(MODEL_PATH)
scaler = brain['scaler']
pca = brain['pca']
kmeans = brain['kmeans']

try:
    ser = serial.Serial(PORT, BAUD, timeout=0.1)
    time.sleep(2)
    
    # DYNAMIC CALIBRATION
    print("\nCALIBRATING... DO NOT TOUCH PLANT")
    baseline_samples = []
    
    while len(baseline_samples) < 40:
        ser.write(b'G')
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        
        if "," in line:
            try:
                c_val = float(line.split(",")[0])
                baseline_samples.append(c_val)
            except:
                continue
    
    AIR_BASELINE = np.mean(baseline_samples)
    print(f"Air Baseline: {AIR_BASELINE:.2f} | SYSTEM LIVE")

    # --- CONTINUOUS STREAMING ---
    while True:
        raw_values = []
        temps = []
        
        # ✅ UPDATED INNER LOOP WITH FAILSAFE
        ser.reset_input_buffer()  # Clear old junk data
        
        attempts = 0  # Start a failsafe counter
        
        # Try to get 20 values, but give up if we fail 50 times (prevents freezing)
        while len(raw_values) < 20 and attempts < 50:
            ser.write(b'G')
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            
            if "," in line:
                try:
                    c_str, t_str = line.split(",")
                    raw_values.append(int(float(c_str)))
                    temps.append(float(t_str))
                except:
                    pass
            
            attempts += 1
            time.sleep(0.01)  # Give Arduino time to respond

        # If Arduino didn't send anything usable
        if len(raw_values) == 0:
            print("\n[WARNING] Arduino skipped a beat. Recovering...")
            time.sleep(0.5)
            continue

        # --- PROCESSING ---
        avg_raw = np.mean(raw_values)
        variance = np.var(raw_values)
        delta_raw = avg_raw - AIR_BASELINE

        # Assemble features
        new_features = np.array([[variance, 0, 0, 0, avg_raw, np.mean(temps)]])
        scaled_feat = scaler.transform(new_features)
        pca_coords = pca.transform(scaled_feat)
        cluster_id = kmeans.predict(pca_coords)[0]

        # HYBRID LOGIC
        if delta_raw < 35:
            final_tree = 0
            growth_val = 0.0
        elif 35 <= delta_raw < 160:
            final_tree = 1
            growth_val = (delta_raw / 160.0)
        else:
            final_tree = 2
            growth_val = 1.0

        # Send to TouchDesigner
        OSC_CLIENT.send_message("/dna/cluster", int(final_tree))
        OSC_CLIENT.send_message("/dna/growth", float(growth_val))
        
        print(f"\rLIVE: Tree {final_tree} | Growth {growth_val:.2f} | Delta {delta_raw:+.1f}     ", end="")

except KeyboardInterrupt:
    print("\nClosing...")

finally:
    ser.close()