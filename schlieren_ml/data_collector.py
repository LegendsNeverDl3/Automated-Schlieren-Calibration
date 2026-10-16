import os
import sys
import cv2
import time
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# Import Duet controller from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from duet_controller import DuetController

# --- CONFIGURATION ---
DUET_IP = "192.168.137.116"
CAMERA_INDEX = 1

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
RAW_SWEEPS_DIR = os.path.join(DATASET_DIR, "raw_sweeps")
PROCESSED_DIR = os.path.join(DATASET_DIR, "processed")
MANIFEST_PATH = os.path.join(DATASET_DIR, "dataset_manifest.csv")

os.makedirs(RAW_SWEEPS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Visual Labeling Key Map
# Key -> (Quality Score, Regime Index, Regime Name, Display Color)
LABEL_KEY_MAP = {
    ord('1'): (1.0, 1, "Optimal Waves (Sweet Spot)", (0, 255, 0)),
    ord('0'): (0.0, 0, "Under-Cutoff (Too Bright / No Waves)", (0, 165, 255)),
    ord('2'): (0.0, 2, "Over-Cutoff (Too Dark / No Waves)", (0, 0, 255)),
    ord('5'): (0.5, 1, "Faint / Emerging Waves", (0, 255, 255)),
    ord('3'): (0.75, 1, "Good Waves (Near Peak)", (100, 255, 100)),
    ord('4'): (0.25, 0, "Very Faint Waves", (0, 200, 200)),
}

def detect_mirror_and_chamber(cap):
    """Detects mirror center and isolates the acoustic chamber ROI."""
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

def capture_temporal_burst(cap, chamber_bbox, num_frames=10):
    """
    Flushes stale driver queue, grabs N frames, and computes
    Temporal Mean and Temporal Variance maps.
    """
    x1, y1, x2, y2 = chamber_bbox
    frames = []
    
    # Flush stale frames from Windows USB camera buffer
    for _ in range(6):
        cap.grab()
        
    # Capture consecutive frames
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        chamber_crop = gray[y1:y2, x1:x2]
        frames.append(chamber_crop)
        time.sleep(0.015)
        
    if len(frames) < 3:
        raise RuntimeError("Insufficient frames captured in burst.")
        
    stack = np.stack(frames, axis=0).astype(np.float32)
    mean_map = np.mean(stack, axis=0).astype(np.uint8)
    var_map = np.var(stack, axis=0)
    var_norm = cv2.normalize(var_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return mean_map, var_norm

def run_interactive_labeled_sweep(duet, cap, chamber_bbox, mirror_geom, start_x, end_x, step_x, condition_tag="default", sound_on=True):
    """
    Moves the motor step-by-step. At each step, displays the live chamber image
    and pauses for human-in-the-loop visual labeling (1=Optimal, 0=Under, 2=Over, etc.).
    """
    cx, cy, rad = mirror_geom
    x1, y1, x2, y2 = chamber_bbox
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    
    sound_str = "sound_on" if sound_on else "sound_off"
    sweep_id = f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{condition_tag}_{sound_str}"
    sweep_folder = os.path.join(PROCESSED_DIR, sweep_id)
    os.makedirs(sweep_folder, exist_ok=True)
    
    total_steps = int(round((end_x - start_x) / step_x)) + 1
    
    print("\n" + "="*70)
    print("--- INTERACTIVE HUMAN-LABELED SWEEP ---")
    print(f"  Sweep ID:      {sweep_id}")
    print(f"  Range:         {start_x:.3f} mm -> {end_x:.3f} mm (Step: {step_x:.3f} mm, ~{total_steps} steps)")
    print(f"  Condition:     {condition_tag}")
    print(f"  Transducers:   {'ON' if sound_on else 'OFF'}")
    print("="*70)
    print("Visual Labeling Controls (press key in camera window):")
    print("  [1] : Optimal Waves (Sweet Spot!) -> Quality = 1.0, Regime = Optimal")
    print("  [0] : Under-Cutoff (Too Bright / No Waves) -> Quality = 0.0, Regime = Under-Cutoff")
    print("  [2] : Over-Cutoff (Too Dark / No Waves) -> Quality = 0.0, Regime = Over-Cutoff")
    print("  [5] : Faint / Emerging Waves -> Quality = 0.5, Regime = Optimal")
    print("  [3] : Good Waves (Near Peak) -> Quality = 0.75, Regime = Optimal")
    print("  [Q] : Stop sweep early & save all labeled data so far")
    print("="*70)
    input("Press [ENTER] when ready to start...")
    
    print(f"\n[Sweep] Moving motor to start position: {start_x:.3f} mm...")
    duet.move_x_absolute(start_x)
    duet.wait_for_move()
    time.sleep(0.8)
    
    window_name = "Interactive Schlieren Labeler"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1100, 750)
    
    records = []
    curr_x = start_x
    step_idx = 0
    last_label_name = "None"
    
    while curr_x <= end_x + 1e-4:
        # 1. Move motor
        duet.move_x_absolute(curr_x)
        duet.wait_for_move()
        
        # 2. Settle vibration
        time.sleep(0.25)
        
        # 3. Capture burst
        mean_img, var_img = capture_temporal_burst(cap, chamber_bbox, num_frames=10)
        
        # Apply CLAHE to zoom view to help see faint waves clearly
        enhanced_chamber = clahe.apply(mean_img)
        
        # Grab current full camera frame for context
        ret, full_frame = cap.read()
        if not ret: full_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 4. Construct rich visual labeling GUI
        # Left: Full frame with mirror and chamber boxes
        ctx = full_frame.copy()
        cv2.circle(ctx, (cx, cy), rad, (255, 120, 0), 2)
        cv2.rectangle(ctx, (x1, y1), (x2, y2), (255, 255, 0), 2)
        ctx_resized = cv2.resize(ctx, (540, 420))
        
        # Right: High-resolution Acoustic Chamber view (Raw Mean + CLAHE Enhanced side-by-side)
        chamber_bgr = cv2.cvtColor(mean_img, cv2.COLOR_GRAY2BGR)
        enhanced_bgr = cv2.cvtColor(enhanced_chamber, cv2.COLOR_GRAY2BGR)
        
        cv2.putText(chamber_bgr, "Raw Mean", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(enhanced_bgr, "Enhanced (Contrast Boost)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        zoom_pair = np.hstack([cv2.resize(chamber_bgr, (265, 420)), cv2.resize(enhanced_bgr, (265, 420))])
        
        main_panel = np.hstack([ctx_resized, zoom_pair])
        h_main, w_main = main_panel.shape[:2]
        
        # Top Header HUD
        header = np.zeros((80, w_main, 3), dtype=np.uint8)
        cv2.putText(header, f"STEP {step_idx+1}/{total_steps} | X = {curr_x:.3f} mm", (15, 32), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        cv2.putText(header, f"Condition: {condition_tag} | Last Label: {last_label_name}", (15, 62), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                    
        # Bottom Instructions HUD
        footer = np.zeros((90, w_main, 3), dtype=np.uint8)
        cv2.putText(footer, "PRESS KEY TO LABEL THIS STEP:", (15, 26), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(footer, "[1] Optimal Waves (1.0)  |  [0] Under-Cutoff / Too Bright (0.0)", (15, 52), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
        cv2.putText(footer, "[2] Over-Cutoff / Too Dark (0.0)  |  [5] Faint Waves (0.5)  |  [Q] Exit", (15, 76), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
                    
        display = np.vstack([header, main_panel, footer])
        cv2.imshow(window_name, display)
        
        # 5. Labeling: Auto-label if sound is OFF, or Wait for Human Keypress if sound is ON
        if not sound_on:
            q_score, reg_label, reg_name = 0.0, 0, "No-Sound-Negative"
            last_label_name = reg_name
            print(f"Step {step_idx+1}/{total_steps} (X = {curr_x:.3f} mm) -> Auto-labeled [No-Sound-Negative]")
            cv2.waitKey(80)  # Brief visual preview, then auto-advance
        else:
            print(f"\nStep {step_idx+1}/{total_steps} (X = {curr_x:.3f} mm) -> Waiting for label key (1, 0, 2, 5, or Q)...")
            while True:
                key = cv2.waitKey(0) & 0xFF
                
                if key in LABEL_KEY_MAP:
                    q_score, reg_label, reg_name, _ = LABEL_KEY_MAP[key]
                    last_label_name = reg_name
                    print(f"  -> Labeled as: {reg_name} (Quality = {q_score})")
                    break
                elif key in (ord('q'), ord('Q'), 27):
                    print("\n[Sweep] Early exit requested by user.")
                    break
                    
            if key in (ord('q'), ord('Q'), 27):
                break
            
        # 6. Save images & record metadata
        sample_id = f"{sweep_id}_step_{step_idx:03d}"
        mean_rel = os.path.relpath(os.path.join(sweep_folder, f"{sample_id}_mean.png"), DATASET_DIR)
        var_rel = os.path.relpath(os.path.join(sweep_folder, f"{sample_id}_var.png"), DATASET_DIR)
        
        cv2.imwrite(os.path.join(DATASET_DIR, mean_rel), mean_img)
        cv2.imwrite(os.path.join(DATASET_DIR, var_rel), var_img)
        
        records.append({
            "sample_id": sample_id,
            "sweep_id": sweep_id,
            "condition": condition_tag,
            "sound_on": int(sound_on),
            "motor_x": round(curr_x, 3),
            "quality_score": q_score,
            "regime_label": reg_label,
            "regime_name": reg_name,
            "mean_path": mean_rel.replace("\\", "/"),
            "var_path": var_rel.replace("\\", "/")
        })
        
        curr_x += step_x
        step_idx += 1
        
    cv2.destroyAllWindows()
    
    if not records:
        print("[Notice] No samples recorded.")
        return
        
    # Append to dataset manifest
    new_df = pd.DataFrame(records)
    if os.path.exists(MANIFEST_PATH):
        existing_df = pd.read_csv(MANIFEST_PATH)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
        
    combined_df.to_csv(MANIFEST_PATH, index=False)
    print("\n" + "="*70)
    print(f"SUCCESS! Saved {len(records)} human-verified samples to {MANIFEST_PATH}!")
    print(f"Total dataset size: {len(combined_df)} samples.")
    print("Label summary for this sweep:")
    print(new_df["regime_name"].value_counts())
    print("="*70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Human-in-the-Loop Schlieren ML Data Collector")
    parser.add_argument("--start", type=float, default=48.60, help="Start X position (mm)")
    parser.add_argument("--end", type=float, default=50.20, help="End X position (mm)")
    parser.add_argument("--step", type=float, default=0.03, help="Step size (mm)")
    parser.add_argument("--condition", type=str, default="normal_light", help="Condition tag")
    parser.add_argument("--sound", choices=["on", "off"], default="on", help="Sound state")
    args = parser.parse_args()
    
    print(f"Connecting to Duet 2 at {DUET_IP}...")
    duet = DuetController(DUET_IP)
    
    print("Opening Camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open camera {CAMERA_INDEX}.")
        sys.exit(1)
        
    try:
        mirror_geom, chamber_bbox = detect_mirror_and_chamber(cap)
        run_interactive_labeled_sweep(
            duet, cap, chamber_bbox, mirror_geom,
            args.start, args.end, args.step,
            condition_tag=args.condition,
            sound_on=(args.sound == "on")
        )
    finally:
        cap.release()
        print("Camera released.")
