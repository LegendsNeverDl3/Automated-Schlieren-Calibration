import os
import sys
import cv2
import time
import argparse
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime

# Import Duet controller from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from duet_controller import DuetController

# --- CONFIGURATION ---
DUET_IP = "192.168.137.65"
CAMERA_INDEX = 1
PRESET_X_POSITION = 49.400

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
RAW_SWEEPS_DIR = os.path.join(DATASET_DIR, "raw_sweeps")
PROCESSED_DIR = os.path.join(DATASET_DIR, "processed")
MANIFEST_PATH = os.path.join(DATASET_DIR, "dataset_manifest.csv")

os.makedirs(RAW_SWEEPS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

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
    
    x1 = int(cx - rad * 0.58)
    x2 = int(cx + rad * 0.58)
    y1 = int(cy - rad * 0.15)
    y2 = int(cy + rad * 0.80)
    
    print(f"[Optics] Mirror Center: ({cx}, {cy}), Radius: {rad}")
    print(f"[Optics] Chamber ROI: X=[{x1}, {x2}], Y=[{y1}, {y2}] ({x2-x1}x{y2-y1} px)")
    return (cx, cy, rad), (x1, y1, x2, y2)

def capture_temporal_burst(cap, chamber_bbox, center_x, num_frames=10):
    """Captures N frames and computes Temporal Mean & Temporal Variance maps."""
    x1, y1, x2, y2 = chamber_bbox
    frames = []
    
    for _ in range(2): cap.read()  # Flush buffer
    
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
    
    # Normalize variance map to 0-255 uint8 for saving
    var_norm = cv2.normalize(var_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Compute classical Sobel-X score for comparison
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    echamber = clahe.apply(mean_map)
    sx = cv2.Sobel(echamber, cv2.CV_64F, 1, 0, ksize=3)
    mid_rel = center_x - x1
    sx[:, max(0, mid_rel - 18) : min(sx.shape[1], mid_rel + 18)] = 0
    classical_score = float(np.var(sx))
    
    return mean_map, var_norm, classical_score

def run_automated_sweep(duet, cap, chamber_bbox, mirror_geom, start_x, end_x, step_x):
    """Executes a calibrated motor sweep, recording video bursts at each position."""
    cx, cy, rad = mirror_geom
    sweep_id = f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sweep_folder = os.path.join(PROCESSED_DIR, sweep_id)
    os.makedirs(sweep_folder, exist_ok=True)
    
    print(f"\n[Sweep] Starting automated collection: {sweep_id}")
    print(f"[Sweep] Moving to start position: {start_x:.3f} mm...")
    duet.move_x_absolute(start_x)
    duet.wait_for_move()
    time.sleep(1.0)
    
    records = []
    curr_x = start_x
    step_idx = 0
    
    cv2.namedWindow("Data Collector Live", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Data Collector Live", 800, 600)
    
    while curr_x <= end_x + 1e-4:
        duet.move_x_absolute(curr_x)
        duet.wait_for_move()
        time.sleep(0.2)
        
        sample_id = f"{sweep_id}_step_{step_idx:03d}"
        mean_img, var_img, classic_score = capture_temporal_burst(cap, chamber_bbox, cx, num_frames=10)
        
        mean_rel_path = os.path.relpath(os.path.join(sweep_folder, f"{sample_id}_mean.png"), DATASET_DIR)
        var_rel_path = os.path.relpath(os.path.join(sweep_folder, f"{sample_id}_var.png"), DATASET_DIR)
        
        cv2.imwrite(os.path.join(DATASET_DIR, mean_rel_path), mean_img)
        cv2.imwrite(os.path.join(DATASET_DIR, var_rel_path), var_img)
        
        records.append({
            "sample_id": sample_id,
            "sweep_id": sweep_id,
            "motor_x": round(curr_x, 3),
            "classical_score": round(classic_score, 1),
            "mean_path": mean_rel_path.replace("\\", "/"),
            "var_path": var_rel_path.replace("\\", "/")
        })
        
        print(f"  Step {step_idx:02d} | X={curr_x:.3f} mm | Classic Score: {classic_score:.1f}")
        
        # Display preview
        preview = np.hstack([mean_img, var_img])
        cv2.putText(preview, f"X: {curr_x:.3f} mm", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("Data Collector Live", preview)
        cv2.waitKey(1)
        
        curr_x += step_x
        step_idx += 1
        
    cv2.destroyAllWindows()
    
    # Post-Sweep Sweet Spot Annotation
    print("\n--- SWEEP COMPLETE ---")
    best_classic = max(records, key=lambda r: r["classical_score"])
    print(f"Classical score peak was at X = {best_classic['motor_x']:.3f} mm (Score: {best_classic['classical_score']:.1f})")
    
    sweet_spot_input = input(f"Enter ground-truth sweet spot X (Press ENTER for {best_classic['motor_x']:.3f} mm): ").strip()
    sweet_spot_x = float(sweet_spot_input) if sweet_spot_input else best_classic["motor_x"]
    
    print(f"Anchoring ground-truth Gaussian Quality curve to X* = {sweet_spot_x:.3f} mm...")
    
    # Calculate Quality score and Cutoff Regime labels
    sigma = 0.12  # Bell-curve width in mm
    for r in records:
        dist = abs(r["motor_x"] - sweet_spot_x)
        # Gaussian Quality in [0.0, 1.0]
        q_score = float(np.exp(-(dist**2) / (2 * (sigma**2))))
        r["quality_score"] = round(q_score, 4)
        
        # Regime Label: 0=Under-Cutoff, 1=Optimal, 2=Over-Cutoff
        if r["motor_x"] < sweet_spot_x - 0.08:
            r["regime_label"] = 0
            r["regime_name"] = "Under-Cutoff"
        elif r["motor_x"] > sweet_spot_x + 0.08:
            r["regime_label"] = 2
            r["regime_name"] = "Over-Cutoff"
        else:
            r["regime_label"] = 1
            r["regime_name"] = "Optimal"
            
    # Save/Append to manifest.csv
    new_df = pd.DataFrame(records)
    if os.path.exists(MANIFEST_PATH):
        existing_df = pd.read_csv(MANIFEST_PATH)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
        
    combined_df.to_csv(MANIFEST_PATH, index=False)
    print(f"Saved {len(records)} samples to {MANIFEST_PATH}! (Total in dataset: {len(combined_df)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Schlieren ML Automated Sweep Data Collector")
    parser.add_argument("--start", type=float, default=48.60, help="Start X position (mm)")
    parser.add_argument("--end", type=float, default=50.20, help="End X position (mm)")
    parser.add_argument("--step", type=float, default=0.03, help="Step size (mm)")
    args = parser.parse_args()
    
    print(f"Connecting to Duet 2 at {DUET_IP}...")
    duet = DuetController(DUET_IP)
    
    print("Opening Camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        sys.exit(1)
        
    try:
        mirror_geom, chamber_bbox = detect_mirror_and_chamber(cap)
        run_automated_sweep(duet, cap, chamber_bbox, mirror_geom, args.start, args.end, args.step)
    finally:
        cap.release()
        print("Camera released.")
