import os
import sys
import cv2
import time
import torch
import numpy as np
from collections import deque

from model import SpatioTemporalWaveNet

# --- CONFIGURATION ---
CAMERA_INDEX = 1
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_model.pth")
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

def run_live_evaluation():
    """Runs real-time ML evaluation on live camera feed without commanding motor movements."""
    print("Initializing Live ML Wave Evaluator (Passive Diagnostic Mode)...")
    
    if not os.path.exists(MODEL_PATH):
        print(f"\n[Notice] No trained weights found at: {MODEL_PATH}")
        print("Using randomly-initialized model for pipeline demonstration.")
        print("Run 'python train.py' once you collect data to evaluate trained weights!\n")
        checkpoint = None
    else:
        print(f"Loading trained model checkpoint from: {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        
    model = SpatioTemporalWaveNet()
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open camera {CAMERA_INDEX}.")
        sys.exit(1)
        
    window_name = "Live ML Schlieren Evaluator (Passive)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 960)
    
    mirror_geom, chamber_bbox = detect_mirror_and_chamber(cap)
    if chamber_bbox is None:
        print("Error: Could not detect mirror. Ensure camera view is unobstructed.")
        cap.release()
        return
        
    cx, cy, rad = mirror_geom
    x1, y1, x2, y2 = chamber_bbox
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    
    buffer = deque(maxlen=8)
    
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
            
            # Prepare tensor for ML inference with CLAHE contrast normalization
            m_res = cv2.resize(echamber, (128, 128)).astype(np.float32) / 255.0
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
        cv2.rectangle(overlay, (10, 10), (450, 115), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, display, 0.35, 0, display)
        cv2.rectangle(display, (10, 10), (450, 115), (120, 120, 120), 1)
        
        regime_text = REGIME_NAMES[ml_regime_idx]
        regime_color = REGIME_COLORS[ml_regime_idx]
        
        cv2.putText(display, f"ML QUALITY SCORE : {ml_quality:.3f} / 1.0", (20, 38), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(display, f"REGIME: {regime_text} ({ml_confidence:.1f}%)", (20, 68), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, regime_color, 2)
        cv2.putText(display, f"CLASSICAL SCORE  : {classic_score:.1f}", (20, 98), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        
        cv2.imshow(window_name, display)
        key = cv2.waitKey(20)
        if key in (ord('q'), ord('Q'), 27):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("Evaluator exited cleanly.")

if __name__ == "__main__":
    run_live_evaluation()
