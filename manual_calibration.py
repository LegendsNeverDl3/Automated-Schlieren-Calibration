import cv2
import numpy as np
import time
from collections import deque
from duet_controller import DuetController

# --- CONFIGURATION ---
DUET_IP = "192.168.137.29"
CAMERA_INDEX = 1           
PRESET_X_POSITION = 49.30

# Stepping Presets
STEP_PRESETS = {
    ord('1'): 0.010,
    ord('2'): 0.020,
    ord('3'): 0.050,
    ord('4'): 0.100,
}
DEFAULT_STEP_SIZE = 0.020
COARSE_STEP_SIZE = 0.100

FRAMES_TO_AVERAGE = 5
FLUSH_FRAMES = 2

print(f"Connecting to Duet 2 at {DUET_IP}...")
duet = DuetController(DUET_IP)

print("Opening Camera...")
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit(1)

# Initialize Resizable OpenCV Window
WINDOW_NAME = "Manual Schlieren Calibration & Debugger"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1280, 960)

# --- AUTOMATIC MIRROR & ACOUSTIC CHAMBER DETECTION ---
print("Detecting mirror geometry & acoustic chamber bounds...")
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
    
    # Strictly isolate the chamber: between towers and below crossbar
    chamber_x1 = int(center_x - radius * 0.58)
    chamber_x2 = int(center_x + radius * 0.58)
    chamber_y1 = int(center_y - radius * 0.15)
    chamber_y2 = int(center_y + radius * 0.80)
    
    print(f"Mirror Center: ({center_x}, {center_y}), Radius: {radius}")
    print(f"Acoustic Chamber ROI: X=[{chamber_x1}, {chamber_x2}], Y=[{chamber_y1}, {chamber_y2}]")
else:
    print("Warning: Could not automatically detect mirror geometry. Exiting.")
    exit(1)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

def get_current_x_coord():
    """Reads current X coordinate from Duet status."""
    coords = duet.get_position()
    if coords and len(coords) > 0:
        return coords[0]
    return PRESET_X_POSITION

def run_manual_calibration():
    print("\n=======================================================")
    print("--- STARTING MANUAL SCHLIEREN CALIBRATION / DEBUGGER ---")
    print("=======================================================")
    print("Controls (Active in Camera Window):")
    print("  [Left Arrow]   : Step -X by active step size")
    print("  [Right Arrow]  : Step +X by active step size")
    print("  [Down Arrow]   : Coarse step -0.100 mm")
    print("  [Up Arrow]     : Coarse step +0.100 mm")
    print("  [1, 2, 3, 4]   : Set step size (1=0.01, 2=0.02, 3=0.05, 4=0.10 mm)")
    print("  [H]            : Return to Preset Home (49.400 mm)")
    print("  [R]            : Reset Highest Peak score tracker")
    print("  [Q or ESC]     : Exit")
    print("=======================================================\n")
    
    # 1. Startup auto-center to preset
    current_x = get_current_x_coord()
    if abs(current_x - PRESET_X_POSITION) > 0.05:
        print(f"Auto-centering to preset position {PRESET_X_POSITION:.3f} mm...")
        duet.move_x_absolute(PRESET_X_POSITION)
        duet.wait_for_move()
        current_x = PRESET_X_POSITION
        time.sleep(0.5)
        
    active_step = DEFAULT_STEP_SIZE
    peak_score = 0.0
    peak_x = current_x
    
    history = deque(maxlen=FRAMES_TO_AVERAGE)
    
    while True:
        # 1. Capture live frame
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        history.append(gray)
        
        # 2. Fast running temporal average (heat plume attenuation)
        avg_gray = np.mean(history, axis=0).astype(np.uint8)
        
        # 3. Extract Acoustic Chamber
        chamber = avg_gray[chamber_y1:chamber_y2, chamber_x1:chamber_x2]
        
        # 4. CLAHE contrast enhancement
        echamber = clahe.apply(chamber)
        
        # 5. Sobel-X for vertical standing wave antinodes
        sx = cv2.Sobel(echamber, cv2.CV_64F, 1, 0, ksize=3)
        
        # 6. Neutralize vertical scratch line in math
        mid_rel = center_x - chamber_x1
        sx[:, max(0, mid_rel - 18) : min(sx.shape[1], mid_rel + 18)] = 0
        
        # 7. Compute Wave Score
        live_score = float(np.var(sx))
        
        if live_score > peak_score:
            peak_score = live_score
            peak_x = current_x
            
        # 8. Render HUD on raw crisp unblurred frame
        display = cv2.cvtColor(avg_gray, cv2.COLOR_GRAY2BGR)
        
        # Blue mirror circle
        cv2.circle(display, (center_x, center_y), radius, (255, 120, 0), 1)
        
        # Cyan Acoustic Chamber Box
        cv2.rectangle(display, (chamber_x1, chamber_y1), (chamber_x2, chamber_y2), (255, 255, 0), 2)
        
        # Draw wave contours inside chamber
        abs_sx = cv2.convertScaleAbs(sx)
        _, wave_thresh = cv2.threshold(abs_sx, 65, 255, cv2.THRESH_BINARY)
        wave_contours, _ = cv2.findContours(wave_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in wave_contours:
            if cv2.contourArea(cnt) > 5:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                cv2.rectangle(display, (chamber_x1 + bx, chamber_y1 + by), 
                              (chamber_x1 + bx + bw, chamber_y1 + by + bh), (0, 255, 0), 1)
                
        # --- HUD OVERLAYS (COMPACT & SEMI-TRANSPARENT) ---
        h_disp, w_disp = display.shape[:2]
        overlay = display.copy()
        
        # Compact Top-Left Diagnostics box
        cv2.rectangle(overlay, (8, 8), (350, 85), (20, 20, 20), -1)
        
        # Ultra-Compact Top-Right Controls Legend box
        legend_w = 245
        legend_h = 75
        cv2.rectangle(overlay, (w_disp - legend_w - 8, 8), (w_disp - 8, 8 + legend_h), (20, 20, 20), -1)
        
        # Blend overlay for semi-transparency (can see right through the boxes)
        cv2.addWeighted(overlay, 0.60, display, 0.40, 0, display)
        
        # Subtle borders
        cv2.rectangle(display, (8, 8), (350, 85), (90, 90, 90), 1)
        cv2.rectangle(display, (w_disp - legend_w - 8, 8), (w_disp - 8, 8 + legend_h), (90, 90, 90), 1)
        
        # Left Diagnostics Text
        cv2.putText(display, f"X: {current_x:.3f} mm | Score: {live_score:.1f}", (16, 32), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(display, f"Peak: {peak_score:.1f} (at {peak_x:.3f} mm)", (16, 62), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 215, 255), 1)
        
        # Right Controls Legend Text (Slim & Unobtrusive)
        cv2.putText(display, f"Step: {active_step:.3f} mm [1-4]", (w_disp - legend_w + 4, 28), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 0), 1)
        cv2.putText(display, "<-/->: Nudge | Up/Dn: 0.1mm", (w_disp - legend_w + 4, 48), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1)
        cv2.putText(display, "H: Home | R: Reset | Q: Quit", (w_disp - legend_w + 4, 68), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 180, 255), 1)
        
        cv2.imshow(WINDOW_NAME, display)
        
        # 9. Handle Keypresses
        # waitKeyEx handles arrow keys on Windows (Left=2424832, Right=2555904, Up=2490368, Down=2621440)
        # Standard waitKey codes: Left=81/2424832, Right=83/2555904
        key = cv2.waitKeyEx(20)
        
        if key == -1:
            continue
            
        # Exit (q, Q, ESC)
        if key in (ord('q'), ord('Q'), 27):
            print("\nExiting manual calibration...")
            break
            
        # Step Size Presets (1, 2, 3, 4)
        if key in STEP_PRESETS:
            active_step = STEP_PRESETS[key]
            print(f"[Controls] Active step size set to: {active_step:.3f} mm")
            
        # Reset Peak Tracker (r, R)
        elif key in (ord('r'), ord('R')):
            peak_score = live_score
            peak_x = current_x
            print(f"[Tracker] Peak reset to current position: X={peak_x:.3f} mm, Score={peak_score:.1f}")
            
        # Home to Preset (h, H)
        elif key in (ord('h'), ord('H')):
            print(f"[Motion] Moving to Home preset {PRESET_X_POSITION:.3f} mm...")
            duet.move_x_absolute(PRESET_X_POSITION)
            duet.wait_for_move()
            current_x = PRESET_X_POSITION
            print(f"[Motion] Arrived at X={current_x:.3f} mm")
            
        # Left Arrow (Step Negative X)
        elif key in (2424832, ord('a'), ord('A'), 81):
            target_x = current_x - active_step
            print(f"[Motion] Stepping Left (-{active_step:.3f} mm) -> Target: {target_x:.3f} mm")
            duet.move_x_absolute(target_x)
            duet.wait_for_move()
            current_x = target_x
            
        # Right Arrow (Step Positive X)
        elif key in (2555904, ord('d'), ord('D'), 83):
            target_x = current_x + active_step
            print(f"[Motion] Stepping Right (+{active_step:.3f} mm) -> Target: {target_x:.3f} mm")
            duet.move_x_absolute(target_x)
            duet.wait_for_move()
            current_x = target_x
            
        # Down Arrow (Coarse Negative X)
        elif key in (2621440, ord('s'), ord('S'), 84):
            target_x = current_x - COARSE_STEP_SIZE
            print(f"[Motion] Coarse Step Down (-{COARSE_STEP_SIZE:.3f} mm) -> Target: {target_x:.3f} mm")
            duet.move_x_absolute(target_x)
            duet.wait_for_move()
            current_x = target_x
            
        # Up Arrow (Coarse Positive X)
        elif key in (2490368, ord('w'), ord('W'), 82):
            target_x = current_x + COARSE_STEP_SIZE
            print(f"[Motion] Coarse Step Up (+{COARSE_STEP_SIZE:.3f} mm) -> Target: {target_x:.3f} mm")
            duet.move_x_absolute(target_x)
            duet.wait_for_move()
            current_x = target_x

if __name__ == "__main__":
    try:
        run_manual_calibration()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released. Exited cleanly.")
