import serial
import time
import numpy as np
from pythonosc import udp_client

# --- CONFIG ---
PORT = 'COM3'
BAUD = 115200
OSC_CLIENT = udp_client.SimpleUDPClient("127.0.0.1", 7000)

print("Starting Biometric Brain (Constraint Mode)...")

try:
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
        
        # Capture 20 samples per frame for instant calculation
        while len(raw_values) < 20:
            ser.write(b'G')
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if "," in line:
                try:
                    c_str = line.split(",")[0]
                    raw_values.append(int(float(c_str)))
                except: continue

        # 1. CALCULATE CORE FEATURES
        avg_raw = np.mean(raw_values)
        variance = np.var(raw_values)
        delta_raw = avg_raw - AIR_BASELINE

        # 2. GROWTH LOGIC (0.0 to 1.0 based strictly on pressure)
        if delta_raw < 35:
            growth_val = 0.0
        else:
            # Map pressure to a smooth 0.0 -> 1.0 curve
            growth_val = min((delta_raw / 160.0), 1.0)

        # 3. MOISTURE CONSTRAINTS (The 3-State Color Logic)
        # We send floats (0.0, 0.5, 1.0) so TouchDesigner can smoothly transition colors
        
        # Constraint A: WET (Sensor maxes out, signal goes totally flat/zero variance)
        if delta_raw > 250 and variance < 1.0:
            moisture_state = 1.0   # 1.0 = Wet Color (Blue)
            state_name = "WET   "
            
        # Constraint B: DRY (High noise/bounciness from dry skin gaps)
        elif variance > 40.0:
            moisture_state = 0.0   # 0.0 = Dry Color (Terracotta/Brown)
            state_name = "DRY   "
            
        # Constraint C: NORMAL (Medium pressure, solid but not perfectly flat connection)
        else:
            moisture_state = 0.5   # 0.5 = Normal Color (Green)
            state_name = "NORMAL"

        # 4. SEND TO TOUCHDESIGNER
        # Notice we are sending floats for BOTH variables now!
        OSC_CLIENT.send_message("/dna/cluster", float(moisture_state)) 
        OSC_CLIENT.send_message("/dna/growth", float(growth_val))
        
        print(f"\rLIVE: [{state_name}] Color: {moisture_state} | Growth: {growth_val:.2f} | Delta: {delta_raw:+.1f} | Var: {variance:.1f}", end="")
        time.sleep(0.05) # Keep loop stable

except KeyboardInterrupt:
    print("\nClosing...")
finally:
    if 'ser' in locals():
        ser.close()