import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split

class SchlierenDataset(Dataset):
    """
    PyTorch Dataset for 2-channel Spatio-Temporal Schlieren images.
    
    Loads:
      - Channel 0: Temporal Mean image (normalized to [0, 1])
      - Channel 1: Temporal Variance map (normalized to [0, 1])
      - Label 1: Continuous Quality Score Q in [0.0, 1.0] (float32)
      - Label 2: Cutoff Regime class index in {0: Under, 1: Optimal, 2: Over} (int64)
    """
    def __init__(self, manifest_df, base_dir="", is_train=True, img_size=(128, 128)):
        self.df = manifest_df.reset_index(drop=True)
        self.base_dir = base_dir
        self.is_train = is_train
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        mean_path = os.path.join(self.base_dir, row["mean_path"])
        var_path = os.path.join(self.base_dir, row["var_path"])
        
        # Load grayscale images
        mean_img = cv2.imread(mean_path, cv2.IMREAD_GRAYSCALE)
        var_img = cv2.imread(var_path, cv2.IMREAD_GRAYSCALE)
        
        if mean_img is None:
            raise FileNotFoundError(f"Could not load mean image at: {mean_path}")
        if var_img is None:
            raise FileNotFoundError(f"Could not load var image at: {var_path}")
            
        # Resize to model standard input size (128x128)
        if (mean_img.shape[1], mean_img.shape[0]) != self.img_size:
            mean_img = cv2.resize(mean_img, self.img_size, interpolation=cv2.INTER_AREA)
            var_img = cv2.resize(var_img, self.img_size, interpolation=cv2.INTER_AREA)
            
        # Strip out global DC background illumination via CLAHE local contrast equalization
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        mean_clahe = clahe.apply(mean_img)
        
        mean_f = mean_clahe.astype(np.float32) / 255.0
        var_f = var_img.astype(np.float32) / 255.0
        
        # Training Augmentations
        if self.is_train:
            # 1. Subtle translations (simulate mechanical positioning jitter)
            shift_x = np.random.randint(-6, 7)
            shift_y = np.random.randint(-4, 5)
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            mean_f = cv2.warpAffine(mean_f, M, self.img_size, borderMode=cv2.BORDER_REFLECT)
            var_f = cv2.warpAffine(var_f, M, self.img_size, borderMode=cv2.BORDER_REFLECT)
            
            # 2. Aggressive illumination invariance (forces AI to ignore overall brightness)
            scale = np.random.uniform(0.65, 1.35)
            bias = np.random.uniform(-0.15, 0.15)
            mean_f = np.clip(mean_f * scale + bias, 0.0, 1.0)
            
            # 3. Additive Gaussian noise
            noise = np.random.normal(0, 0.015, self.img_size).astype(np.float32)
            mean_f = np.clip(mean_f + noise, 0.0, 1.0)
            
        # Stack into 2-channel tensor (2, H, W)
        stacked = np.stack([mean_f, var_f], axis=0)
        tensor_x = torch.from_numpy(stacked)
        
        quality = torch.tensor(float(row.get("quality_score", 0.0)), dtype=torch.float32)
        regime = torch.tensor(int(row.get("regime_label", 0)), dtype=torch.long)
        
        metadata = {
            "motor_x": float(row.get("motor_x", 0.0)),
            "sample_id": str(row.get("sample_id", f"sample_{idx}"))
        }
        
        return tensor_x, quality, regime, metadata

def get_dataloaders(manifest_csv_path, batch_size=16, val_ratio=0.2, seed=42):
    """Factory to create train and validation DataLoaders from a manifest CSV."""
    if not os.path.exists(manifest_csv_path):
        raise FileNotFoundError(f"Manifest CSV not found at: {manifest_csv_path}")
        
    df = pd.read_csv(manifest_csv_path)
    base_dir = os.path.dirname(manifest_csv_path)
    
    np.random.seed(seed)
    n_total = len(df)
    n_val = int(n_total * val_ratio)
    indices = np.random.permutation(n_total)
    
    train_indices = indices[n_val:]
    val_indices = indices[:n_val]
    
    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)
    
    train_dataset = SchlierenDataset(train_df, base_dir=base_dir, is_train=True)
    val_dataset = SchlierenDataset(val_df, base_dir=base_dir, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader
