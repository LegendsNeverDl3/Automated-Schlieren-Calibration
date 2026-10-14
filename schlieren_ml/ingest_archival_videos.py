import os
import cv2
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

VIDEOS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "schlieren_cv_videos"))
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
PROCESSED_DIR = os.path.join(DATASET_DIR, "processed")
MANIFEST_PATH = os.path.join(DATASET_DIR, "dataset_manifest.csv")

os.makedirs(PROCESSED_DIR, exist_ok=True)

def detect_chamber_from_video(cap):
    """Detects mirror and chamber ROI from the first 30 frames of a video."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, 5)
    ret, frame = cap.read()
    if not ret:
        return None, None
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Fallback to standard center crop if thresholding fails
        h, w = gray.shape
        cx, cy, rad = w // 2, h // 2, min(w, h) // 3
    else:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        cx, cy = x + w // 2, y + h // 2
        rad = min(w, h) // 2 - 10
        
    x1 = int(cx - rad * 0.58)
    x2 = int(cx + rad * 0.58)
    y1 = int(cy - rad * 0.15)
    y2 = int(cy + rad * 0.80)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return (cx, cy, rad), (x1, y1, x2, y2)

def process_video_file(video_path, burst_len=10, stride=15, default_regime=1, quality_multiplier=1.0):
    """Extracts temporal bursts (Mean + Variance) from a recorded video."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = os.path.join(PROCESSED_DIR, f"archival_{video_name}")
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Warning] Could not open video: {video_path}")
        return []
        
    mirror_geom, chamber_bbox = detect_chamber_from_video(cap)
    if chamber_bbox is None:
        print(f"[Warning] Could not detect chamber in: {video_name}")
        cap.release()
        return []
        
    cx, cy, rad = mirror_geom
    x1, y1, x2, y2 = chamber_bbox
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n[Ingest] Processing '{video_name}' ({total_frames} frames)...")
    
    records = []
    frame_idx = 0
    buffer = []
    burst_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        crop = gray[y1:y2, x1:x2]
        buffer.append(crop)
        
        if len(buffer) == burst_len:
            # Process burst
            stack = np.stack(buffer, axis=0).astype(np.float32)
            mean_map = np.mean(stack, axis=0).astype(np.uint8)
            var_map = np.var(stack, axis=0)
            var_norm = cv2.normalize(var_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Classical score
            echamber = clahe.apply(mean_map)
            sx = cv2.Sobel(echamber, cv2.CV_64F, 1, 0, ksize=3)
            mid_rel = cx - x1
            sx[:, max(0, mid_rel - 18) : min(sx.shape[1], mid_rel + 18)] = 0
            classical_score = float(np.var(sx))
            
            sample_id = f"{video_name}_b{burst_idx:04d}"
            mean_rel = os.path.relpath(os.path.join(output_folder, f"{sample_id}_mean.png"), DATASET_DIR)
            var_rel = os.path.relpath(os.path.join(output_folder, f"{sample_id}_var.png"), DATASET_DIR)
            
            cv2.imwrite(os.path.join(DATASET_DIR, mean_rel), mean_map)
            cv2.imwrite(os.path.join(DATASET_DIR, var_rel), var_norm)
            
            # Normalize classical score roughly to [0, 1] as initial pseudo-label
            pseudo_q = min(1.0, (classical_score / 2000.0) * quality_multiplier)
            
            records.append({
                "sample_id": sample_id,
                "sweep_id": f"archival_{video_name}",
                "motor_x": 49.400,  # Nominal reference
                "classical_score": round(classical_score, 1),
                "quality_score": round(pseudo_q, 4),
                "regime_label": default_regime,
                "regime_name": "Optimal" if default_regime == 1 else ("Under-Cutoff" if default_regime == 0 else "Over-Cutoff"),
                "mean_path": mean_rel.replace("\\", "/"),
                "var_path": var_rel.replace("\\", "/")
            })
            
            burst_idx += 1
            # Slide window by stride
            buffer = buffer[stride:] if stride < burst_len else []
            
        frame_idx += 1
        
    cap.release()
    print(f"  Extracted {len(records)} temporal bursts from '{video_name}'.")
    return records

def ingest_all_videos():
    """Ingests all MP4 videos in schlieren_cv_videos/ into the dataset manifest."""
    if not os.path.exists(VIDEOS_DIR):
        print(f"[Error] Videos directory not found at: {VIDEOS_DIR}")
        return
        
    mp4_files = [f for f in os.listdir(VIDEOS_DIR) if f.lower().endswith(".mp4")]
    if not mp4_files:
        print(f"No MP4 files found in {VIDEOS_DIR}")
        return
        
    print(f"Found {len(mp4_files)} archival videos in {VIDEOS_DIR}:")
    for f in mp4_files:
        print(f"  - {f}")
        
    all_records = []
    for f in mp4_files:
        full_path = os.path.join(VIDEOS_DIR, f)
        records = process_video_file(full_path, burst_len=10, stride=15)
        all_records.extend(records)
        
    if not all_records:
        print("No records extracted.")
        return
        
    new_df = pd.DataFrame(all_records)
    if os.path.exists(MANIFEST_PATH):
        existing_df = pd.read_csv(MANIFEST_PATH)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
        
    combined_df.to_csv(MANIFEST_PATH, index=False)
    print(f"\n[Success] Ingested {len(all_records)} archival bursts into {MANIFEST_PATH}!")
    print(f"Total dataset size is now: {len(combined_df)} samples.")

if __name__ == "__main__":
    ingest_all_videos()
