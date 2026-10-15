import cv2
import numpy as np
import time
from collections import deque
from duet_controller import DuetController

# --- CONFIGURATION ---
DUET_IP = "192.168.137.243"  # Your Duet 2 IP
CAMERA_INDEX = 1           

PRESET_X_POSITION = 49.30   # Base center position

# Contracting Sweep (Multi-Pass Back-and-Forth) Settings
PASS_1_SPAN = 0.60   # Pass 1: Forward sweep across ±0.60mm
PASS_1_STEP = 0.05   # Pass 1: Coarse step size (0.05mm)

PASS_2_SPAN = 0.16   # Pass 2: Reverse sweep across ±0.16mm around peak
PASS_2_STEP = 0.02   # Pass 2: Fine step size (0.02mm)

PASS_3_SPAN = 0.05   # Pass 3: Micro sweep across ±0.05mm
PASS_3_STEP = 0.01   # Pass 3: Micro step size (0.01mm)

FRAMES_TO_AVERAGE = 10     
FLUSH_FRAMES = 5           

print(f"Connecting to Duet 2 at {DUET_IP}...")
duet = DuetController(DUET_IP)

print("Opening Camera...")
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit(1)

# Initialize a resizable window once
cv2.namedWindow("Acoustic Wave Contracting Scanner", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Acoustic Wave Contracting Scanner", 1280, 960)

# --- AUTOMATIC MIRROR & ACOUSTIC CHAMBER DETECTION ---
print("Detecting mirror geometry & acoustic levitation chamber...")
for _ in range(10): cap.read() 
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    largest_cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_cnt)
    
    center_x = x + (w // 2)
    center_y = y + (h // 2)
    radius = min(w, h) // 2 - 10
    
    # Define the Acoustic Chamber: strictly between transducer towers and below crossbar
    chamber_x1 = int(center_x - radius * 0.58)
    chamber_x2 = int(center_x + radius * 0.58)
    chamber_y1 = int(center_y - radius * 0.15)
    chamber_y2 = int(center_y + radius * 0.80)
    
    print(f"Mirror Center: ({center_x}, {center_y}), Radius: {radius}")
    print(f"Acoustic Chamber ROI: X=[{chamber_x1}, {chamber_x2}], Y=[{chamber_y1}, {chamber_y2}]")
else:
    print("Warning: Could not automatically detect the mirror.")
    exit(1)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

def evaluate_acoustic_chamber(pass_name, current_x):
    """Samples frames, filters out heat, and scores vertical standing wave fringes in the chamber."""
    for _ in range(FLUSH_FRAMES): cap.read()
        
    history = deque(maxlen=FRAMES_TO_AVERAGE)
    last_frame = None
    for _ in range(FRAMES_TO_AVERAGE):
        ret, frame = cap.read()
        if not ret: break
        last_frame = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        history.append(gray)
        
    # 1. Temporal average to eliminate transient thermal plumes
    avg_gray = np.mean(history, axis=0).astype(np.uint8)
    
    # 2. Extract strictly the Acoustic Chamber directly (crisp, no inpaint blurring)
    chamber = avg_gray[chamber_y1:chamber_y2, chamber_x1:chamber_x2]
    
    # 3. CLAHE enhancement on the chamber
    echamber = clahe.apply(chamber)
    
    # 4. Sobel-X detects vertical standing wave fringes
    sx = cv2.Sobel(echamber, cv2.CV_64F, 1, 0, ksize=3)
    
    # 5. Mathematically isolate and zero out the lens scratch from the score calculation ONLY
    mid_rel = center_x - chamber_x1
    sx[:, max(0, mid_rel - 18) : min(sx.shape[1], mid_rel + 18)] = 0
    
    # 6. Calculate variance of acoustic wave fringes (completely unaffected by the scratch)
    acoustic_score = np.var(sx)
    
    # --- VISUALIZATION ---
    # Display the 100% crisp, natural optical frame (no blurring)
    display = cv2.cvtColor(avg_gray, cv2.COLOR_GRAY2BGR)
    
    # Blue circle for full mirror
    cv2.circle(display, (center_x, center_y), radius, (255, 120, 0), 1)
    
    # Cyan rectangle around Acoustic Chamber
    cv2.rectangle(display, (chamber_x1, chamber_y1), (chamber_x2, chamber_y2), (255, 255, 0), 2)
    
    # Extract wave contours to outline standing wave fringes in green
    abs_sx = cv2.convertScaleAbs(sx)
    _, wave_thresh = cv2.threshold(abs_sx, 65, 255, cv2.THRESH_BINARY)
    wave_contours, _ = cv2.findContours(wave_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in wave_contours:
        if cv2.contourArea(cnt) > 5:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            cv2.rectangle(display, (chamber_x1 + bx, chamber_y1 + by), 
                          (chamber_x1 + bx + bw, chamber_y1 + by + bh), (0, 255, 0), 1)
            
    cv2.putText(display, f"{pass_name} | X: {current_x:.3f} mm", (10, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(display, f"Wave Contrast Score: {acoustic_score:.1f}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    
    cv2.imshow("Acoustic Wave Contracting Scanner", display)
    cv2.waitKey(1)
    
    return acoustic_score

def execute_sweep_pass(pass_name, start_pos, end_pos, step_size):
    """Executes a single sweep pass from start_pos to end_pos with given step_size."""
    print(f"\n>>> {pass_name}: Sweeping from {start_pos:.3f}mm to {end_pos:.3f}mm (step: {step_size:.3f}mm)...")
    
    duet.move_x_absolute(start_pos)
    duet.wait_for_move()
    time.sleep(1.0)
    
    sweep_data = []
    
    # Determine direction
    is_forward = end_pos >= start_pos
    curr = start_pos
    
    while (curr <= end_pos + 1e-4) if is_forward else (curr >= end_pos - 1e-4):
        duet.move_x_absolute(curr)
        duet.wait_for_move()
        time.sleep(0.18)
        
        score = evaluate_acoustic_chamber(pass_name, curr)
        sweep_data.append((curr, score))
        print(f"  X = {curr:.3f} mm -> Wave Score: {score:.1f}")
        
        curr = (curr + step_size) if is_forward else (curr - abs(step_size))
        
    # Find peak of this pass
    best_pos, best_score = max(sweep_data, key=lambda item: item[1])
    print(f"--- {pass_name} Peak Found at X = {best_pos:.3f} mm (Score: {best_score:.1f}) ---")
    return best_pos, best_score

def optimize_schlieren_contracting():
    print("\n=======================================================")
    print("--- STARTING MULTI-PASS CONTRACTING ACOUSTIC SWEEP ---")
    print("=======================================================")
    
    # Ensure starting near preset
    coords = duet.get_position()
    if coords:
        if abs(coords[0] - PRESET_X_POSITION) > 0.1:
            print(f"Moving to preset starting position ({PRESET_X_POSITION} mm)...")
            duet.move_x_absolute(PRESET_X_POSITION)
            duet.wait_for_move()
            time.sleep(1.0)
    else:
        duet.move_x_absolute(PRESET_X_POSITION)
        duet.wait_for_move()
        time.sleep(1.0)
        
    # --- PASS 1: FORWARD COARSE SWEEP ---
    p1_start = PRESET_X_POSITION - PASS_1_SPAN
    p1_end = PRESET_X_POSITION + PASS_1_SPAN
    peak_1, score_1 = execute_sweep_pass("Pass 1 (Forward Coarse)", p1_start, p1_end, PASS_1_STEP)
    
    # --- PASS 2: REVERSE FINE SWEEP (Narrowed Range) ---
    p2_start = peak_1 + PASS_2_SPAN
    p2_end = peak_1 - PASS_2_SPAN
    peak_2, score_2 = execute_sweep_pass("Pass 2 (Reverse Fine)", p2_start, p2_end, -PASS_2_STEP)
    
    # --- PASS 3: FORWARD MICRO LOCK (Tight Window) ---
    p3_start = peak_2 - PASS_3_SPAN
    p3_end = peak_2 + PASS_3_SPAN
    peak_3, score_3 = execute_sweep_pass("Pass 3 (Micro Precision)", p3_start, p3_end, PASS_3_STEP)
    
    # --- FINAL SETTLE ---
    print("\n=======================================================")
    print(f"🎯 OPTIMAL ALIGNMENT LOCATED: X = {peak_3:.3f} mm (Score: {score_3:.1f})")
    print(f"Moving motor to final precision target...")
    print("=======================================================")
    
    duet.move_x_absolute(peak_3)
    duet.wait_for_move()
    time.sleep(0.5)
    
    # Final display
    evaluate_acoustic_chamber("ALIGNED (FINAL)", peak_3)
    print("System successfully aligned! Waiting 3 seconds before closing.")
    cv2.waitKey(3000)

if __name__ == "__main__":
    try:
        optimize_schlieren_contracting()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nCamera released. Done.")
