import os
import cv2
import argparse
import numpy as np
import pandas as pd

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
MANIFEST_PATH = os.path.join(DATASET_DIR, "dataset_manifest.csv")

REGIME_MAP = {
    0: ("Under-Cutoff", (0, 165, 255)),   # Orange
    1: ("Optimal", (0, 255, 0)),          # Green
    2: ("Over-Cutoff", (0, 0, 255))       # Red
}

def load_manifest():
    if not os.path.exists(MANIFEST_PATH):
        print(f"[Error] Manifest not found at: {MANIFEST_PATH}")
        return None
    return pd.read_csv(MANIFEST_PATH)

def batch_label_by_video():
    """Mode 1: Shows a preview collage of each video and lets you tag all its samples at once."""
    df = load_manifest()
    if df is None: return
    
    videos = df["sweep_id"].unique().tolist()
    print("\n" + "="*65)
    print("--- VIDEO-LEVEL BATCH LABELER ---")
    print(f"Found {len(videos)} unique video sequences across {len(df)} samples.")
    print("For each video, inspect the sample collage and press:")
    print("  [1] : Optimal Waves (Quality = 1.0, Regime = Optimal)")
    print("  [0] : Under-Cutoff / No Waves (Quality = 0.0, Regime = Under-Cutoff)")
    print("  [2] : Over-Cutoff / Too Dark (Quality = 0.0, Regime = Over-Cutoff)")
    print("  [S] : Skip / Leave as-is")
    print("  [Q] : Save & Exit")
    print("="*65 + "\n")
    
    cv2.namedWindow("Video Batch Labeler", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video Batch Labeler", 1024, 600)
    
    for idx, vid_id in enumerate(videos):
        subset = df[df["sweep_id"] == vid_id]
        sample_rows = subset.sample(min(8, len(subset)), random_state=42)
        
        # Build preview collage of sample bursts
        thumbnails = []
        for _, row in sample_rows.iterrows():
            m_path = os.path.join(DATASET_DIR, row["mean_path"])
            v_path = os.path.join(DATASET_DIR, row["var_path"])
            
            m_img = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
            v_img = cv2.imread(v_path, cv2.IMREAD_GRAYSCALE)
            
            if m_img is None or v_img is None:
                continue
                
            m_res = cv2.resize(m_img, (128, 128))
            v_res = cv2.resize(v_img, (128, 128))
            pair = np.hstack([m_res, v_res])
            thumbnails.append(pair)
            
        if not thumbnails:
            continue
            
        # Arrange into a 2x4 grid
        while len(thumbnails) < 8:
            thumbnails.append(np.zeros_like(thumbnails[0]))
            
        row1 = np.hstack(thumbnails[:4])
        row2 = np.hstack(thumbnails[4:8])
        collage = np.vstack([row1, row2])
        collage_bgr = cv2.cvtColor(collage, cv2.COLOR_GRAY2BGR)
        
        # Overlay instructions
        h_col, w_col = collage_bgr.shape[:2]
        header = np.zeros((70, w_col, 3), dtype=np.uint8)
        cv2.putText(header, f"Video {idx+1}/{len(videos)}: {vid_id} ({len(subset)} samples)", 
                    (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(header, "Press: [1]=Optimal | [0]=Under-Cutoff/No Waves | [2]=Over-Cutoff | [S]=Skip", 
                    (15, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1)
                    
        display = np.vstack([header, collage_bgr])
        cv2.imshow("Video Batch Labeler", display)
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('1'):
                df.loc[df["sweep_id"] == vid_id, "quality_score"] = 1.0
                df.loc[df["sweep_id"] == vid_id, "regime_label"] = 1
                df.loc[df["sweep_id"] == vid_id, "regime_name"] = "Optimal"
                print(f"[{idx+1}/{len(videos)}] Marked '{vid_id}' ({len(subset)} samples) as OPTIMAL (Quality=1.0)")
                break
            elif key == ord('0'):
                df.loc[df["sweep_id"] == vid_id, "quality_score"] = 0.0
                df.loc[df["sweep_id"] == vid_id, "regime_label"] = 0
                df.loc[df["sweep_id"] == vid_id, "regime_name"] = "Under-Cutoff"
                print(f"[{idx+1}/{len(videos)}] Marked '{vid_id}' ({len(subset)} samples) as UNDER-CUTOFF / NO WAVES (Quality=0.0)")
                break
            elif key == ord('2'):
                df.loc[df["sweep_id"] == vid_id, "quality_score"] = 0.0
                df.loc[df["sweep_id"] == vid_id, "regime_label"] = 2
                df.loc[df["sweep_id"] == vid_id, "regime_name"] = "Over-Cutoff"
                print(f"[{idx+1}/{len(videos)}] Marked '{vid_id}' ({len(subset)} samples) as OVER-CUTOFF (Quality=0.0)")
                break
            elif key in (ord('s'), ord('S'), 32):  # Space or S
                print(f"[{idx+1}/{len(videos)}] Skipped '{vid_id}'.")
                break
            elif key in (ord('q'), ord('Q'), 27):
                df.to_csv(MANIFEST_PATH, index=False)
                cv2.destroyAllWindows()
                print("\nSaved updates and exited.")
                return
                
    df.to_csv(MANIFEST_PATH, index=False)
    cv2.destroyAllWindows()
    print("\n" + "="*65)
    print(f"SUCCESS! All batch labels saved to: {MANIFEST_PATH}")
    print("Summary of labeled dataset:")
    print(df["regime_name"].value_counts())
    print("="*65)

def interactive_sample_reviewer():
    """Mode 2: Fine-grained inspection of each individual burst sample."""
    df = load_manifest()
    if df is None: return
    
    cv2.namedWindow("Sample Reviewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sample Reviewer", 800, 500)
    
    print("\n--- SAMPLE-BY-SAMPLE REVIEWER ---")
    print("Controls: [1]=Optimal | [0]=Under-Cutoff | [2]=Over-Cutoff | [D]=Delete | [Left/Right]=Prev/Next | [Q]=Quit\n")
    
    idx = 0
    while 0 <= idx < len(df):
        row = df.iloc[idx]
        m_path = os.path.join(DATASET_DIR, row["mean_path"])
        v_path = os.path.join(DATASET_DIR, row["var_path"])
        
        m_img = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
        v_img = cv2.imread(v_path, cv2.IMREAD_GRAYSCALE)
        
        if m_img is None or v_img is None:
            idx += 1
            continue
            
        m_lg = cv2.resize(m_img, (256, 256))
        v_lg = cv2.resize(v_img, (256, 256))
        
        # Add labels
        cv2.putText(m_lg, "Temporal Mean", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(v_lg, "Temporal Variance", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        pair = np.hstack([m_lg, v_lg])
        display = cv2.cvtColor(pair, cv2.COLOR_GRAY2BGR)
        
        # Top HUD
        hud = np.zeros((60, display.shape[1], 3), dtype=np.uint8)
        reg_name, color = REGIME_MAP.get(int(row["regime_label"]), ("Unknown", (255, 255, 255)))
        q_val = float(row["quality_score"])
        
        cv2.putText(hud, f"Sample {idx+1}/{len(df)}: {row['sample_id']}", (10, 22), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(hud, f"Quality: {q_val:.2f} | Regime: {reg_name}", (10, 48), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                    
        canvas = np.vstack([hud, display])
        cv2.imshow("Sample Reviewer", canvas)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('1'):
            df.at[idx, "quality_score"] = 1.0
            df.at[idx, "regime_label"] = 1
            df.at[idx, "regime_name"] = "Optimal"
            idx += 1
        elif key == ord('0'):
            df.at[idx, "quality_score"] = 0.0
            df.at[idx, "regime_label"] = 0
            df.at[idx, "regime_name"] = "Under-Cutoff"
            idx += 1
        elif key == ord('2'):
            df.at[idx, "quality_score"] = 0.0
            df.at[idx, "regime_label"] = 2
            df.at[idx, "regime_name"] = "Over-Cutoff"
            idx += 1
        elif key in (2555904, 83, ord('d'), ord('D'), 32):  # Right arrow / space
            idx += 1
        elif key in (2424832, 81, ord('a'), ord('A'), 8):    # Left arrow / backspace
            idx = max(0, idx - 1)
        elif key in (ord('q'), ord('Q'), 27):
            break
            
    df.to_csv(MANIFEST_PATH, index=False)
    cv2.destroyAllWindows()
    print(f"\nManifest saved to: {MANIFEST_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Schlieren ML Dataset Labeler")
    parser.add_argument("--mode", choices=["batch", "sample"], default="batch", 
                        help="Choose 'batch' (by video) or 'sample' (individual bursts)")
    args = parser.parse_args()
    
    if args.mode == "batch":
        batch_label_by_video()
    else:
        interactive_sample_reviewer()
