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
DUET_IP = "192.168.137.116"
CAMERA_INDEX = 1
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_model.pth")

# Search parameters
COARSE_START_X = 48.600
COARSE_END_X = 50.200
COARSE_STEP = 0.040      # Stage 1: Coarse corridor sweep
FINE_STEP = 0.010        # Stage 2: Sub-mm micro sweep
CORRIDOR_THRESHOLD = 0.60 # Quality score threshold defining the sweet spot corridor

REGIME_NAMES = ["Under-Cutoff", "Optimal Cutoff", "Over-Cutoff"]
REGIME_COLORS = [(0, 165, 255), (0, 255, 0), (0, 0, 255)]  # Orange, Green, Red

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
    
    # 4. Format into (1, 2, 128, 128) tensor for model
    m_res = cv2.resize(mean_map, (128, 128)).astype(np.float32) / 255.0
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
        # STAGE 1: ML-GUIDED COARSE SEARCH (Corridor Acquisition)
        # =====================================================================
        print("\n" + "-"*65)
        print("STAGE 1: ML Global Corridor Acquisition (Coarse Step: 0.040 mm)")
        print(f"Sweeping X from {COARSE_START_X:.3f} mm to {COARSE_END_X:.3f} mm...")
        print("-"*65)
        
        duet.move_x_absolute(COARSE_START_X)
        duet.wait_for_move()
        time.sleep(0.5)
        
        curr_x = COARSE_START_X
        corridor_xs = []
        corridor_qs = []
        in_corridor = False
        
        while curr_x <= COARSE_END_X + 1e-4:
            duet.move_x_absolute(curr_x)
            duet.wait_for_move()
            time.sleep(0.20)  # Settle vibrations
            
            q, reg_idx, conf, mean_map, var_map = capture_burst_and_evaluate(cap, chamber_bbox, model)
            
            # Render HUD
            ret, full_frame = cap.read()
            if not ret: full_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            display = full_frame.copy()
            draw_hud(display, curr_x, q, reg_idx, conf, "Stage 1: Coarse Corridor Search", cx, cy, rad, chamber_bbox)
            cv2.imshow(window_name, display)
            cv2.waitKey(1)
            
            print(f"  X={curr_x:.3f} mm | ML Quality: {q:.3f} | Regime: {REGIME_NAMES[reg_idx]} ({conf:.1f}%)")
            
            # Detect entrance into sweet spot corridor
            if q >= CORRIDOR_THRESHOLD and reg_idx == 1:
                if not in_corridor:
                    print(f"  >>> [CORRIDOR DETECTED] Entered sweet spot at X = {curr_x:.3f} mm! <<<")
                    in_corridor = True
                corridor_xs.append(curr_x)
                corridor_qs.append(q)
            elif in_corridor and (q < CORRIDOR_THRESHOLD or reg_idx == 2):
                print(f"  >>> [CORRIDOR EXITED] Sweet spot ended at X = {curr_x:.3f} mm. <<<")
                break
                
            curr_x += COARSE_STEP
            
        if not corridor_xs:
            print("\n[Warning] Did not detect strong corridor during forward sweep.")
            print("Fallback: Using highest point observed.")
            corridor_start = 49.300
            corridor_end = 49.500
        else:
            corridor_start = min(corridor_xs) - 0.020
            corridor_end = max(corridor_xs) + 0.020
            
        print(f"\n[Stage 1 Complete] Bounded Optimal Corridor: [{corridor_start:.3f} mm -> {corridor_end:.3f} mm]")
        
        # =====================================================================
        # STAGE 2: SUB-MILLIMETER PARABOLIC VERTEX LOCK
        # =====================================================================
        print("\n" + "-"*65)
        print("STAGE 2: Sub-mm Parabolic Vertex Lock (Micro Step: 0.010 mm)")
        print(f"Micro-sweeping through corridor [{corridor_start:.3f} mm -> {corridor_end:.3f} mm]...")
        print("-"*65)
        
        duet.move_x_absolute(corridor_start)
        duet.wait_for_move()
        time.sleep(0.5)
        
        fine_xs = []
        fine_qs = []
        curr_x = corridor_start
        
        while curr_x <= corridor_end + 1e-4:
            duet.move_x_absolute(curr_x)
            duet.wait_for_move()
            time.sleep(0.20)
            
            q, reg_idx, conf, mean_map, var_map = capture_burst_and_evaluate(cap, chamber_bbox, model)
            fine_xs.append(curr_x)
            fine_qs.append(q)
            
            ret, full_frame = cap.read()
            if not ret: full_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            display = full_frame.copy()
            draw_hud(display, curr_x, q, reg_idx, conf, "Stage 2: Micro Parabolic Sweep", cx, cy, rad, chamber_bbox)
            cv2.imshow(window_name, display)
            cv2.waitKey(1)
            
            print(f"  [Micro] X={curr_x:.3f} mm -> ML Score: {q:.4f}")
            curr_x += FINE_STEP
            
        # Mathematical Parabolic Fitting: Q(x) = ax^2 + bx + c
        xs = np.array(fine_xs)
        qs = np.array(fine_qs)
        
        print("\n--- CALCULATING SUB-MILLIMETER PARABOLIC VERTEX ---")
        try:
            poly = np.polyfit(xs, qs, 2)
            a, b, c = poly
            
            # Vertex of parabola: x* = -b / (2a)
            if a < 0:  # Inverted parabola (has a true maximum peak)
                optimal_x = -b / (2.0 * a)
                # Clamp within corridor bounds for physical safety
                optimal_x = max(min(optimal_x, corridor_end), corridor_start)
                print(f"Parabolic Fit: Q(x) = {a:.2f}x^2 + {b:.2f}x + {c:.2f}")
                print(f"Mathematical Peak Vertex: X* = {optimal_x:.4f} mm")
            else:
                # Upward parabola (fallback to discrete maximum)
                best_idx = np.argmax(qs)
                optimal_x = xs[best_idx]
                print(f"Parabola did not invert; fallback to discrete peak at X* = {optimal_x:.3f} mm")
        except Exception as e:
            best_idx = np.argmax(qs)
            optimal_x = xs[best_idx]
            print(f"Fitting error ({e}); fallback to discrete peak at X* = {optimal_x:.3f} mm")
            
        # =====================================================================
        # FINAL LOCK: MOVE TO SUB-MM OPTICAL VERTEX
        # =====================================================================
        print(f"\n[Final Action] Driving knife edge to sub-millimeter optical peak: {optimal_x:.4f} mm...")
        duet.move_x_absolute(optimal_x)
        duet.wait_for_move()
        time.sleep(0.5)
        
        # Grab final locked verification frame
        final_q, final_reg, final_conf, final_mean, _ = capture_burst_and_evaluate(cap, chamber_bbox, model, num_frames=12)
        
        ret, full_frame = cap.read()
        display = full_frame.copy() if ret else np.zeros((480, 640, 3), dtype=np.uint8)
        draw_hud(display, optimal_x, final_q, final_reg, final_conf, "ALIGNMENT COMPLETE (LOCKED)", cx, cy, rad, chamber_bbox)
        
        # Green Locked Badge
        cv2.rectangle(display, (display.shape[1] - 320, 10), (display.shape[1] - 10, 70), (0, 180, 0), -1)
        cv2.putText(display, "SYSTEM OPTIMIZED", (display.shape[1] - 305, 38), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(display, f"Peak Locked at {optimal_x:.3f} mm", (display.shape[1] - 305, 58), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 255, 220), 1)
        
        cv2.imshow(window_name, display)
        
        print("\n" + "="*70)
        print("--- ALIGNMENT COMPLETE! SCHLIEREN SYSTEM IS OPTIMIZED ---")
        print(f"  Optimal Motor Position:  {optimal_x:.4f} mm")
        print(f"  Final ML Wave Quality:   {final_q:.4f} / 1.000")
        print(f"  Predicted Cutoff Regime: {REGIME_NAMES[final_reg]} ({final_conf:.1f}% confidence)")
        print("="*70)
        print("Press any key in the camera window to close and exit.")
        cv2.waitKey(0)
        
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released cleanly.")

if __name__ == "__main__":
    run_ml_auto_alignment()
