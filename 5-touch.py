import serial
import time
import csv

# --- ARCHITECT CONFIG ---
PORT = 'COM3'       
BAUD = 115200
SAMPLES = 50        
TRIALS = 5          
DELAY = 0.025       

try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)
    print("--- BIOMETRIC ARCHITECT V6: THERMAL + CAPACITIVE ENGINE ---")

    while True:
        name = input("\nParticipant Name: ")
        scenario = input("Scenario (Silent/Hum): ")
        label = f"{name}_{scenario}"

        # 1. CALIBRATE AIR
        print("\n[1] CALIBRATING AIR (Don't touch!)...")
        ser.reset_input_buffer()
        air_vals = []
        while len(air_vals) < 20:
            ser.write(b'G') 
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            
            # --- SAFETY FIX START ---
            if line and "," in line:
                try:
                    parts = line.split(",")
                    if len(parts) == 2:
                        c_str, t_str = parts
                        air_vals.append(int(float(c_str)))
                        print(f"\rProgress: {len(air_vals)}/20", end="")
                        time.sleep(0.01)
                except (ValueError, IndexError):
                    continue
            # --- SAFETY FIX END ---
        
        avg_air = sum(air_vals) / len(air_vals)
        print(f"\nAir Baseline: {avg_air:.2f}")

        # 2. DATA COLLECTION
        for i in range(1, TRIALS + 1):
            print(f"\n--- Trial {i}/{TRIALS} ---")
            print(">>> Action: TOUCH SENSOR and hold steady...")
            time.sleep(1.0) 
            input(">>> Press ENTER to Record.")
            
            ser.reset_input_buffer()
            deltas = []
            raws = []
            temps = [] 
            
            print(f"Syncing {SAMPLES} points...")
            while len(deltas) < SAMPLES:
                ser.write(b'G')
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                
                # --- SAFETY FIX START ---
                if line and "," in line:
                    try:
                        parts = line.split(",")
                        if len(parts) == 2:
                            c_str, t_str = parts
                            r = int(float(c_str))
                            t = float(t_str)
                            
                            raws.append(r)
                            temps.append(t)
                            deltas.append(r - avg_air)
                            print(f"\rCaptured: {len(deltas)}/{SAMPLES}", end="")
                            time.sleep(DELAY)
                    except (ValueError, IndexError):
                        continue
                # --- SAFETY FIX END ---

            # 3. FEATURE EXTRACTION
            mean_raw = sum(raws) / len(raws)
            mean_delta = sum(deltas) / len(deltas)
            avg_temp = sum(temps) / len(temps) 
            
            variance = sum((x - mean_delta)**2 for x in deltas) / len(deltas)
            jaggedness = sum(abs(deltas[j] - deltas[j-1]) for j in range(1, len(deltas)))
            total_slope = (deltas[-1] - deltas[0]) / SAMPLES
            sig_range = max(deltas) - min(deltas)

            print(f"\nCapture Complete!")
            print(f" -> Avg Raw: {mean_raw:.2f} | Avg Temp: {avg_temp:.2f}°C")
            print(f" -> Jaggedness: {jaggedness:.2f} | Variance: {variance:.2f}")

            # 4. SAVE TO CSV
            features = [
                round(variance, 4), 
                round(total_slope, 4), 
                round(jaggedness, 2), 
                round(sig_range, 2), 
                round(mean_raw, 2),
                round(avg_temp, 2) 
            ]
            
            with open('biometric_architect_data.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([label] + deltas + features)

        if input("\nRecord more? (y/n): ").lower() != 'y': 
            break

finally:
    if 'ser' in locals(): 
        ser.close()
        print("\nSerial Closed.")