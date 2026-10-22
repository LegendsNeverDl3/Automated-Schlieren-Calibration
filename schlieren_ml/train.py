import os
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import SpatioTemporalWaveNet
from dataset import get_dataloaders

# --- HYPERPARAMETERS ---
MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "dataset", "dataset_manifest.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
CURVES_PLOT_PATH = os.path.join(MODELS_DIR, "training_curves.png")

os.makedirs(MODELS_DIR, exist_ok=True)

def train_model(epochs=50, batch_size=16, lr=1e-3, val_split=0.2, model_version="v1"):
    use_v2 = (model_version.lower() in ["v2", "2"])
    version_label = "Version 2 (CLAHE Normalized)" if use_v2 else "Version 1 (Raw Baseline)"
    
    if use_v2:
        best_model_target = os.path.join(MODELS_DIR, "best_model_v2.pth")
    else:
        best_model_target = os.path.join(MODELS_DIR, "best_model_v1.pth")
        
    print("\n" + "="*70)
    print(f"--- TRAINING SPATIO-TEMPORAL WAVENET [{version_label.upper()}] ---")
    print(f"  Target Checkpoint : {best_model_target}")
    print(f"  Preprocessing     : {'CLAHE Enabled' if use_v2 else 'Raw Linear Inputs'}")
    print("="*70 + "\n")
    
    if not os.path.exists(MANIFEST_PATH):
        print(f"[Error] Dataset manifest not found at: {MANIFEST_PATH}")
        print("Run 'python data_collector.py' first to collect sweep data!")
        return
        
    print(f"Loading dataset from: {MANIFEST_PATH}")
    train_loader, val_loader = get_dataloaders(
        MANIFEST_PATH, batch_size=batch_size, val_ratio=val_split, use_clahe=use_v2
    )
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # --- HANDLE DATASET SKEW / CLASS IMBALANCE ---
    regime_counts = [0, 0, 0]
    for _, _, r, _ in train_loader.dataset:
        regime_counts[r.item()] += 1
    total_samples = len(train_loader.dataset)
    class_weights = [total_samples / (3.0 * max(1, c)) for c in regime_counts]
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    print(f"Dataset Regime Counts (Train): Under={regime_counts[0]}, Optimal={regime_counts[1]}, Over={regime_counts[2]}")
    print(f"Applied Class Loss Multipliers: Under={class_weights[0]:.2f}x, Optimal={class_weights[1]:.2f}x, Over={class_weights[2]:.2f}x")
    
    model = SpatioTemporalWaveNet().to(device)
    
    mse_criterion = nn.MSELoss()
    ce_criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_mse": [], "val_acc": []}
    
    for epoch in range(1, epochs + 1):
        # 1. Train Step
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_q, batch_r, _ in train_loader:
            batch_x = batch_x.to(device)
            batch_q = batch_q.to(device)
            batch_r = batch_r.to(device)
            
            optimizer.zero_grad()
            pred_q, pred_r = model(batch_x)
            
            # 4x penalty on high-quality samples to combat skew
            weight_mask = 1.0 + 3.0 * (batch_q > 0.40).float()
            loss_mse = torch.mean(weight_mask * (pred_q - batch_q) ** 2)
            loss_ce = ce_criterion(pred_r, batch_r)
            
            loss = loss_mse + 0.5 * loss_ce
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(batch_x)
            
        scheduler.step()
        train_loss /= len(train_loader.dataset)
        
        # 2. Validation Step
        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        correct_regime = 0
        total_val = len(val_loader.dataset)
        
        with torch.no_grad():
            for batch_x, batch_q, batch_r, _ in val_loader:
                batch_x = batch_x.to(device)
                batch_q = batch_q.to(device)
                batch_r = batch_r.to(device)
                
                pred_q, pred_r = model(batch_x)
                
                loss_mse = mse_criterion(pred_q, batch_q)
                loss_ce = ce_criterion(pred_r, batch_r)
                loss = loss_mse + 0.5 * loss_ce
                
                val_loss += loss.item() * len(batch_x)
                val_mse += loss_mse.item() * len(batch_x)
                
                preds_class = torch.argmax(pred_r, dim=1)
                correct_regime += (preds_class == batch_r).sum().item()
                
        val_loss /= total_val
        val_mse /= total_val
        val_acc = (correct_regime / total_val * 100.0) if total_val > 0 else 0.0
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mse"].append(val_mse)
        history["val_acc"].append(val_acc)
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            ckpt_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "val_mse": val_mse,
                "val_acc": val_acc,
                "model_version": "v2" if use_v2 else "v1"
            }
            torch.save(ckpt_dict, best_model_target)
            if not use_v2:
                # Also save to default best_model.pth
                torch.save(ckpt_dict, os.path.join(MODELS_DIR, "best_model.pth"))
            save_star = " [BEST MODEL SAVED]"
        else:
            save_star = ""
            
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Val MSE: {val_mse:.4f} | Regime Acc: {val_acc:.1f}%{save_star}")
        
    print("\n" + "="*60)
    print(f"Training Complete! Best model saved to: {best_model_target}")
    print(f"Final Best Val Loss: {best_val_loss:.4f}")
    
    # Plot Training Curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title(f"Multi-Task Loss ({version_label})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history["val_acc"], label="Regime Accuracy (%)", color="green")
    plt.title(f"Validation Regime Accuracy ({version_label})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(CURVES_PLOT_PATH)
    print(f"Saved training curve plot to: {CURVES_PLOT_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SpatioTemporalWaveNet")
    parser.add_argument("--version", "-v", "--model-version", choices=["v1", "1", "v2", "2"], default="v1",
                        help="Model version to train: 'v1' / '1' (Raw baseline, default) or 'v2' / '2' (CLAHE normalized)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    
    v_key = "v2" if args.version in ["v2", "2"] else "v1"
    train_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, model_version=v_key)
