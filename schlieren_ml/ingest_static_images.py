import os
import cv2
import argparse
import numpy as np
import pandas as pd

IMAGES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "schlieren_cv_images"))
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
PROCESSED_DIR = os.path.join(DATASET_DIR, "processed", "static_images")
MANIFEST_PATH = os.path.join(DATASET_DIR, "dataset_manifest.csv")

os.makedirs(PROCESSED_DIR, exist_ok=True)

def detect_chamber_from_image(img):
    """Detects mirror and acoustic chamber ROI from a static image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
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
    
    return (cx, cy, rad), (x1, y1, x2, y2)

def ingest_static_images(source_dir=IMAGES_DIR):
    """Ingests static image files (.jpg, .png) into the ML dataset."""
    if not os.path.exists(source_dir):
        print(f"[Error] Source directory not found: {source_dir}")
        return
        
    img_files = [f for f in os.listdir(source_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if not img_files:
        print(f"No image files found in {source_dir}")
        return
        
    print(f"\n[Ingest] Ingesting {len(img_files)} static images from {source_dir}...")
    records = []
    
    for f in img_files:
        path = os.path.join(source_dir, f)
        img = cv2.imread(path)
        if img is None:
            continue
            
        mirror_geom, chamber_bbox = detect_chamber_from_image(img)
        cx, cy, rad = mirror_geom
        x1, y1, x2, y2 = chamber_bbox
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        crop = gray[y1:y2, x1:x2]
        
        # Channel 0: Image crop (Mean)
        # Channel 1: For static images with zero time dimension, use blank (zeros)
        var_norm = np.zeros_like(crop, dtype=np.uint8)
        
        name_no_ext = os.path.splitext(f)[0]
        sample_id = f"static_{name_no_ext}"
        
        mean_rel = os.path.relpath(os.path.join(PROCESSED_DIR, f"{sample_id}_mean.png"), DATASET_DIR)
        var_rel = os.path.relpath(os.path.join(PROCESSED_DIR, f"{sample_id}_var.png"), DATASET_DIR)
        
        cv2.imwrite(os.path.join(DATASET_DIR, mean_rel), crop)
        cv2.imwrite(os.path.join(DATASET_DIR, var_rel), var_norm)
        
        # Classical score
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        echamber = clahe.apply(crop)
        sx = cv2.Sobel(echamber, cv2.CV_64F, 1, 0, ksize=3)
        mid_rel = cx - x1
        sx[:, max(0, mid_rel - 18) : min(sx.shape[1], mid_rel + 18)] = 0
        classic_score = float(np.var(sx))
        
        records.append({
            "sample_id": sample_id,
            "sweep_id": "static_images",
            "motor_x": 49.400,
            "classical_score": round(classic_score, 1),
            "quality_score": 0.5,     # Default to neutral until labeled
            "regime_label": 1,        # Default to optimal
            "regime_name": "Optimal",
            "mean_path": mean_rel.replace("\\", "/"),
            "var_path": var_rel.replace("\\", "/")
        })
        print(f"  Processed {f} (Classic Score: {classic_score:.1f})")
        
    if not records:
        return
        
    new_df = pd.DataFrame(records)
    if os.path.exists(MANIFEST_PATH):
        existing_df = pd.read_csv(MANIFEST_PATH)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
        
    combined_df.to_csv(MANIFEST_PATH, index=False)
    print(f"\n[Success] Ingested {len(records)} static images into {MANIFEST_PATH}!")
    print("Run 'python schlieren_ml/labeler.py --mode sample' to quickly label them!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default=IMAGES_DIR, help="Folder containing images")
    args = parser.parse_args()
    ingest_static_images(args.source)
