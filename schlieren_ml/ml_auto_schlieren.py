import os
import sys
import cv2
import time
import torch
import numpy as np
from datetime import datetime

# Ensure both current directory and parent repo directory are in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from duet_controller import DuetController
from model import SpatioTemporalWaveNet

# --- HARDWARE & OPTICS CONFIGURATION ---
DUET_IP = "192.168.137.243"
CAMERA_INDEX = 1
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_model.pth")

# Search parameters
START_X = 48.600
MAX_X = 50.400
STEP_SIZE = 0.020        # 20 micron steps for smooth peak profiling
CORRIDOR_THRESHOLD = 0.55 # Minimum score to consider entering sweet spot
MIN_DROPS_TO_CONFIRM = 3  # Number of consecutive decreasing readings to confirm peak is passed
DROP_TOLERANCE = 0.15     # Total drop from peak required to confirm downward slope

REGIME_NAMES = ["Under-Cutoff", "Optimal Cutoff", "Over-Cutoff"]
REGIME_COLORS = [(0, 165, 255), (0, 255, 0), (0, 0, 255)]  # Orange, Green, Red

def move_x_unidirectional(duet, target_x, overshoot_distance=0.250):
    """
    Drives to target_x always arriving in the forward (+X) direction.
    If moving left (negative direction), retracts past target_x by overshoot_distance,
    then approaches forward (+X). This completely eliminates mechanical leadscrew backlash.
    """
    retract_x = target_x - overshoot_distance
    duet.move_x_absolute(retract_x)
    duet.wait_for_move()
    time.sleep(0.3)
    
    duet.move_x_absolute(target_x)
    duet.wait_for_move()
    time.sleep(0.35)

def detect_mirror_and_chamber(cap):
    """Detects mirror center and isolates the acoustic chamber ROI with boundary safety."""
    for _ in range(10): cap.read()
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to capture from camera during chamber detection.")
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise RuntimeError("No mirror contour detected. Ensure camera is pointed at Schlieren mirror.")
        
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    cx, cy = x + w // 2, y + h // 2
    rad = min(w, h) // 2 - 10
    
    h_f, w_f = gray.shape[:2]
    x1 = max(0, int(cx - rad * 0.58))
    x2 = min(w_f, int(cx + rad * 0.58))
    y1 = max(0, int(cy - rad * 0.15))
    y2 = min(h_f, int(cy + rad * 0.80))
    
    print(f"[Optics] Mirror Center: ({cx}, {cy}), Radius: {rad}")
    print(f"[Optics] Chamber ROI: X=[{x1}, {x2}], Y=[{y1}, {y2}] ({x2-x1}x{y2-y1} px)")
    return (cx, cy, rad), (x1, y1, x2, y2)

def capture_burst_and_evaluate(cap, chamber_bbox, model, num_frames=10):
    """Flushes driver buffers, grabs burst frames, and runs neural network inference."""
    x1, y1, x2, y2 = chamber_bbox
    frames = []
    
    # 1. Flush stale USB camera driver queue
    for _ in range(6): cap.grab()
    
    # 2. Capture consecutive frames
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        chamber_crop = gray[y1:y2, x1:x2]
        frames.append(chamber_crop)
        time.sleep(0.015)
        
    if len(frames) < 3:
        raise RuntimeError("Insufficient frames captured in burst.")
        
    # 3. Compute Mean & Variance maps
    stack = np.stack(frames, axis=0).astype(np.float32)
    mean_map = np.mean(stack, axis=0).astype(np.uint8)
    var_map = np.var(stack, axis=0)
    var_norm = cv2.normalize(var_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 4. Format into (1, 2, 128, 128) tensor for model with CLAHE contrast normalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    mean_clahe = clahe.apply(mean_map)
    m_res = cv2.resize(mean_clahe, (128, 128)).astype(np.float32) / 255.0
    v_res = cv2.resize(var_norm, (128, 128)).astype(np.float32) / 255.0
    tensor_x = torch.from_numpy(np.stack([m_res, v_res], axis=0)).unsqueeze(0)
    
    # 5. Neural Network Inference
    with torch.no_grad():
        pred_q, pred_r = model(tensor_x)
        probs = torch.softmax(pred_r, dim=1)[0]
        quality = float(pred_q[0].item())
        regime_idx = int(torch.argmax(probs).item())
        confidence = float(probs[regime_idx].item()) * 100.0
        
    return quality, regime_idx, confidence, mean_map, var_norm

def draw_hud(display, curr_x, quality, regime_idx, confidence, stage_text, cx, cy, rad, chamber_bbox):
    """Renders semi-transparent diagnostic HUD over live display."""
    x1, y1, x2, y2 = chamber_bbox
    cv2.circle(display, (cx, cy), rad, (255, 120, 0), 2)
    cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 0), 2)
    
    h_disp, w_disp = display.shape[:2]
    overlay = display.copy()
    cv2.rectangle(overlay, (10, 10), (520, 130), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, display, 0.35, 0, display)
    cv2.rectangle(display, (10, 10), (520, 130), (120, 120, 120), 1)
    
    reg_name = REGIME_NAMES[regime_idx]
    reg_color = REGIME_COLORS[regime_idx]
    
    cv2.putText(display, f"STAGE: {stage_text}", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2)
    cv2.putText(display, f"MOTOR X: {curr_x:.3f} mm", (20, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(display, f"ML QUALITY: {quality:.3f} / 1.0", (20, 95), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(display, f"REGIME: {reg_name} ({confidence:.1f}%)", (20, 122), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, reg_color, 2)
                
    # Quality progress bar
    bar_x, bar_y, bar_w, bar_h = 320, 80, 180, 18
    cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
    fill_w = int(bar_w * max(0.0, min(1.0, quality)))
    cv2.rectangle(display, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 255, 0), -1)
    cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)

def run_ml_auto_alignment():
    print("\n" + "="*70)
    print("--- ML-GUIDED CLOSED-LOOP SCHLIEREN AUTO-ALIGNMENT ---")
    print("="*70)
    
    # 1. Load Trained Neural Network
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at: {MODEL_PATH}. Run train.py first!")
        
    print(f"[AI] Loading trained weights from: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model = SpatioTemporalWaveNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("[AI] SpatioTemporalWaveNet ready on CPU.")
    
    # 2. Connect Duet 2
    print(f"[Duet] Connecting to Duet 2 at {DUET_IP}...")
    duet = DuetController(DUET_IP)
    
    # 3. Open Camera
    print("Opening Camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")
        
    window_name = "ML Auto-Schlieren Closed-Loop Alignment"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 850)
    
    try:
        mirror_geom, chamber_bbox = detect_mirror_and_chamber(cap)
        cx, cy, rad = mirror_geom
        
        # =====================================================================
        # FORWARD SWEEP WITH PEAK-AND-PAST NEGATIVE-SLOPE DETECTION
        # =====================================================================
        print("\n" + "-"*65)
        print(f"FORWARD SWEEP: Stepping X from {START_X:.3f} mm to {MAX_X:.3f} mm (Step: {STEP_SIZE:.3f} mm)")
        print("Climbing toward peak, then stepping PAST to confirm negative slope...")
        print("-"*65)
        
        duet.move_x_absolute(START_X)
        duet.wait_for_move()
        time.sleep(0.5)
        
        curr_x = START_X
        all_xs = []
        all_qs = []
        
        peak_q = 0.0
        peak_x = START_X
        prev_q = 0.0
        consecutive_drops = 0
        peak_confirmed = False
        
        while curr_x <= MAX_X + 1e-4:
            duet.move_x_absolute(curr_x)
            duet.wait_for_move()
            time.sleep(0.20)  # Settle mechanical vibrations
            
            q, reg_idx, conf, mean_map, var_map = capture_burst_and_evaluate(cap, chamber_bbox, model)
            all_xs.append(curr_x)
            all_qs.append(q)
            
            # Render HUD
            ret, full_frame = cap.read()
            if not ret: full_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            display = full_frame.copy()
            status_text = "Hunting for Peak..." if consecutive_drops == 0 else f"Confirming Negative Slope ({consecutive_drops}/{MIN_DROPS_TO_CONFIRM})"
            draw_hud(display, curr_x, q, reg_idx, conf, f"Forward Sweep ({status_text})", cx, cy, rad, chamber_bbox)
            cv2.imshow(window_name, display)
            cv2.waitKey(1)
            
            print(f"  X={curr_x:.3f} mm | Quality: {q:.4f} | Regime: {REGIME_NAMES[reg_idx]} ({conf:.1f}%) | Peak: {peak_q:.4f} (at {peak_x:.3f} mm)")
            
            # Track peak and monitor negative slope
            if q > peak_q:
                peak_q = q
                peak_x = curr_x
                consecutive_drops = 0
            elif peak_q >= CORRIDOR_THRESHOLD:
                # We have reached a strong wave signal and are now past the peak candidate
                if q < prev_q:
                    consecutive_drops += 1
                else:
                    consecutive_drops = max(0, consecutive_drops - 1)
                    
                drop_from_peak = peak_q - q
                print(f"    -> [Post-Peak Monitor] Drops: {consecutive_drops}/{MIN_DROPS_TO_CONFIRM} | Drop from peak: -{drop_from_peak:.3f}")
                
                # Confirm negative slope: multiple consecutive drops and dropped by at least DROP_TOLERANCE
                if consecutive_drops >= MIN_DROPS_TO_CONFIRM and drop_from_peak >= DROP_TOLERANCE:
                    print(f"\n>>> [NEGATIVE SLOPE CONFIRMED] Passed the sweet spot! <<<")
                    print(f"    Peak recorded at:  X = {peak_x:.3f} mm (Score: {peak_q:.4f})")
                    print(f"    Current reading:   X = {curr_x:.3f} mm (Score: {q:.4f})")
                    print(f"    Observed {consecutive_drops} consecutive drops over {(curr_x - peak_x):.3f} mm past peak.")
                    peak_confirmed = True
                    break
                    
            prev_q = q
            curr_x += STEP_SIZE
            
        if not peak_confirmed:
            print(f"\n[Notice] Reached sweep boundary. Highest score recorded was at X = {peak_x:.3f} mm.")
            
        # =====================================================================
        # PARABOLIC VERTEX REFINEMENT
        # =====================================================================
        xs_arr = np.array(all_xs)
        qs_arr = np.array(all_qs)
        
        # Window of points around the crest (climb, peak, descent)
        mask = (xs_arr >= peak_x - 0.070) & (xs_arr <= peak_x + 0.070)
        fit_xs = xs_arr[mask]
        fit_qs = qs_arr[mask]
        
        optimal_x = peak_x
        if len(fit_xs) >= 4:
            try:
                poly = np.polyfit(fit_xs, fit_qs, 2)
                a, b, c = poly
                if a < 0:  # Inverted parabola with a true maximum peak
                    vertex_x = -b / (2.0 * a)
                    # Safety check: vertex must be within microns of the discrete peak
                    if abs(vertex_x - peak_x) <= 0.030:
                        optimal_x = vertex_x
                        print(f"\n--- PARABOLIC REFINEMENT ---")
                        print(f"Fit: Q(x) = {a:.2f}x^2 + {b:.2f}x + {c:.2f}")
                        print(f"Discrete Peak: {peak_x:.3f} mm -> Mathematical Vertex: {optimal_x:.4f} mm")
            except Exception as fit_err:
                print(f"[Notice] Parabolic fit fallback to discrete peak ({fit_err})")
                
        # =====================================================================
        # STAGE 2: ANTI-BACKLASH RETURN & CLOSED-LOOP MICRO-POLISH
        # =====================================================================
        print("\n" + "="*70)
        print("STAGE 2: Anti-Backlash Return & Closed-Loop Optical Peak Polish")
        print("="*70)
        
        # Candidate positions around the peak to eliminate any remaining overshoot / backlash
        # Prioritize left-shifted points (-0.06 to +0.02 mm) to completely prevent overshooting
        polish_offsets = [-0.060, -0.040, -0.020, 0.000, +0.020]
        test_positions = [optimal_x + off for off in polish_offsets]
        
        print(f"[Anti-Backlash] Retracting left to start unidirectional forward polish...")
        # Move well past the leftmost candidate to load lead screw in forward (+X) direction
        duet.move_x_absolute(min(test_positions) - 0.250)
        duet.wait_for_move()
        time.sleep(0.4)
        
        best_live_q = -1.0
        best_live_x = optimal_x
        best_live_reg = 1
        best_live_conf = 0.0
        
        print("\n[Micro-Polish] Stepping forward through peak candidates with 0 microns backlash:")
        for tx in sorted(test_positions):
            # Always step forward into candidate
            duet.move_x_absolute(tx)
            duet.wait_for_move()
            time.sleep(0.25)
            
            lq, lreg, lconf, lmean, lvar = capture_burst_and_evaluate(cap, chamber_bbox, model, num_frames=10)
            print(f"  Candidate X={tx:.3f} mm -> Live ML Quality: {lq:.4f} | Regime: {REGIME_NAMES[lreg]} ({lconf:.1f}%)")
            
            # Update HUD
            ret, full_frame = cap.read()
            display = full_frame.copy() if ret else np.zeros((480, 640, 3), dtype=np.uint8)
            draw_hud(display, tx, lq, lreg, lconf, "Stage 2: Closed-Loop Polish", cx, cy, rad, chamber_bbox)
            cv2.imshow(window_name, display)
            cv2.waitKey(1)
            
            if lq > best_live_q:
                best_live_q = lq
                best_live_x = tx
                best_live_reg = lreg
                best_live_conf = lconf
                
        # Drive forward to the confirmed best live position
        print(f"\n[Final Action] Locking knife-edge to best live optical position: X = {best_live_x:.3f} mm...")
        move_x_unidirectional(duet, best_live_x, overshoot_distance=0.200)
        
        # Grab final locked verification frame
        final_q, final_reg, final_conf, final_mean, _ = capture_burst_and_evaluate(cap, chamber_bbox, model, num_frames=12)
        
        print("\n" + "="*70)
        print("--- ALIGNMENT COMPLETE! SCHLIEREN SYSTEM IS OPTIMIZED ---")
        print(f"  Initial Estimated Peak:   {peak_x:.3f} mm")
        print(f"  Final Locked Position:    {best_live_x:.3f} mm")
        print(f"  Correction Applied:       {(best_live_x - peak_x):+.3f} mm")
        print(f"  Final ML Wave Quality:    {final_q:.4f} / 1.000")
        print(f"  Predicted Cutoff Regime:  {REGIME_NAMES[final_reg]} ({final_conf:.1f}% confidence)")
        print("="*70)
        
        ret, full_frame = cap.read()
        display = full_frame.copy() if ret else np.zeros((480, 640, 3), dtype=np.uint8)
        draw_hud(display, best_live_x, final_q, final_reg, final_conf, "ALIGNMENT COMPLETE (LOCKED)", cx, cy, rad, chamber_bbox)
        
        # Green Locked Badge
        cv2.rectangle(display, (display.shape[1] - 340, 10), (display.shape[1] - 10, 75), (0, 180, 0), -1)
        cv2.putText(display, "SYSTEM OPTIMIZED", (display.shape[1] - 325, 38), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(display, f"Peak: {final_q:.3f} at {best_live_x:.3f} mm", (display.shape[1] - 325, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 255, 220), 1)
        
        cv2.imshow(window_name, display)
        print("Press any key in the camera window to close and exit.")
        cv2.waitKey(0)
        
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released cleanly.")

if __name__ == "__main__":
    run_ml_auto_alignment()
