import os
import sys
import cv2
import time
import argparse
import torch
import numpy as np
from collections import deque

from model import SpatioTemporalWaveNet

# --- CONFIGURATION ---
CAMERA_INDEX = 1
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

REGIME_NAMES = ["Under-Cutoff", "Optimal Cutoff", "Over-Cutoff"]
REGIME_COLORS = [(0, 165, 255), (0, 255, 0), (0, 0, 255)]  # Orange, Green, Red

def detect_mirror_and_chamber(cap):
    """Detects mirror and chamber ROI."""
    for _ in range(10): cap.read()
    ret, frame = cap.read()
    if not ret: return None, None
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: return None, None
    
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    cx, cy = x + w // 2, y + h // 2
    rad = min(w, h) // 2 - 10
    
    x1 = int(cx - rad * 0.58)
    x2 = int(cx + rad * 0.58)
    y1 = int(cy - rad * 0.15)
    y2 = int(cy + rad * 0.80)
    
    return (cx, cy, rad), (x1, y1, x2, y2)

def run_live_evaluation(model_version="v1", camera_index=CAMERA_INDEX):
    """Runs real-time ML evaluation on live camera feed without commanding motor movements."""
    use_v2 = (model_version.lower() in ["v2", "2"])
    version_label = "v2 (CLAHE Contrast Normalized)" if use_v2 else "v1 (Raw Baseline, Recommended)"
    
    if use_v2:
        model_path = os.path.join(MODELS_DIR, "best_model_v2.pth")
    else:
        model_path = os.path.join(MODELS_DIR, "best_model_v1.pth")
        if not os.path.exists(model_path):
            model_path = os.path.join(MODELS_DIR, "best_model.pth")
            
    use_clahe = use_v2
    
    print("\n" + "="*70)
    print("--- LIVE ML SCHLIEREN EVALUATOR (PASSIVE DIAGNOSTIC MODE) ---")
    print(f"  Configuration : {version_label}")
    print(f"  Model Path    : {model_path}")
    print(f"  Preprocessing : {'CLAHE Normalized' if use_clahe else 'Raw Inputs (No CLAHE)'}")
    print("="*70 + "\n")
    
    if not os.path.exists(model_path):
        print(f"[Error] Model checkpoint not found at: {model_path}")
        sys.exit(1)
        
    checkpoint = torch.load(model_path, map_location="cpu")
    model = SpatioTemporalWaveNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("[AI] SpatioTemporalWaveNet ready on CPU.")
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}.")
        sys.exit(1)
        
    window_name = f"Live ML Schlieren Evaluator [{model_version.upper()}]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 960)
    
    mirror_geom, chamber_bbox = detect_mirror_and_chamber(cap)
    if chamber_bbox is None:
        print("Error: Could not detect mirror. Ensure camera view is unobstructed.")
        cap.release()
        sys.exit(1)
        
    cx, cy, rad = mirror_geom
    x1, y1, x2, y2 = chamber_bbox
    
    buffer = deque(maxlen=8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    
    print("\n--- Live Evaluator Running ---")
    print("Press 'Q' or ESC in the camera window to exit.\n")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        chamber_crop = gray[y1:y2, x1:x2]
        buffer.append(chamber_crop)
        
        ml_quality = 0.0
        ml_regime_idx = 0
        ml_confidence = 0.0
        classic_score = 0.0
        
        if len(buffer) == buffer.maxlen:
            stack = np.stack(list(buffer), axis=0).astype(np.float32)
            mean_map = np.mean(stack, axis=0).astype(np.uint8)
            var_map = np.var(stack, axis=0)
            var_norm = cv2.normalize(var_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Classical score for comparison
            echamber = clahe.apply(mean_map)
            sx = cv2.Sobel(echamber, cv2.CV_64F, 1, 0, ksize=3)
            mid_rel = cx - x1
            sx[:, max(0, mid_rel - 18) : min(sx.shape[1], mid_rel + 18)] = 0
            classic_score = float(np.var(sx))
            
            # Prepare tensor for ML inference
            if use_clahe:
                mean_in = clahe.apply(mean_map)
            else:
                mean_in = mean_map
                
            m_res = cv2.resize(mean_in, (128, 128)).astype(np.float32) / 255.0
            v_res = cv2.resize(var_norm, (128, 128)).astype(np.float32) / 255.0
            tensor_x = torch.from_numpy(np.stack([m_res, v_res], axis=0)).unsqueeze(0)
            
            t0 = time.time()
            with torch.no_grad():
                pred_q, pred_r = model(tensor_x)
                probs = torch.softmax(pred_r, dim=1)[0]
                ml_quality = float(pred_q[0].item())
                ml_regime_idx = int(torch.argmax(probs).item())
                ml_confidence = float(probs[ml_regime_idx].item()) * 100.0
            inf_time_ms = (time.time() - t0) * 1000.0
            
        # Visual Render
        display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Mirror & Chamber outlines
        cv2.circle(display, (cx, cy), rad, (255, 120, 0), 1)
        cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 0), 2)
        
        # HUD Panel (Semi-transparent)
        overlay = display.copy()
        cv2.rectangle(overlay, (10, 10), (520, 135), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, display, 0.35, 0, display)
        cv2.rectangle(display, (10, 10), (520, 135), (120, 120, 120), 1)
        
        regime_text = REGIME_NAMES[ml_regime_idx]
        regime_color = REGIME_COLORS[ml_regime_idx]
        
        cv2.putText(display, f"VERSION: {version_label}", (20, 32), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 215, 255), 1)
        cv2.putText(display, f"ML QUALITY SCORE : {ml_quality:.3f} / 1.0", (20, 62), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(display, f"REGIME: {regime_text} ({ml_confidence:.1f}%)", (20, 92), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, regime_color, 2)
        cv2.putText(display, f"CLASSICAL SCORE  : {classic_score:.1f}", (20, 122), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        
        cv2.imshow(window_name, display)
        key = cv2.waitKey(20)
        if key in (ord('q'), ord('Q'), 27):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("Evaluator exited cleanly.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Passive ML Wave Evaluator")
    parser.add_argument("--model-version", "--version", "-v", choices=["v1", "1", "v2", "2"], default="v1",
                        help="Model version: 'v1' (Raw baseline, default) or 'v2' (CLAHE normalized)")
    parser.add_argument("--cam", type=int, default=CAMERA_INDEX, help=f"Camera index (default: {CAMERA_INDEX})")
    args = parser.parse_args()
    
    v_key = "v2" if args.model_version in ["v2", "2"] else "v1"
    run_live_evaluation(model_version=v_key, camera_index=args.cam)
