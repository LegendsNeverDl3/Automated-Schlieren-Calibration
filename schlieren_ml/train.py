import os
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
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pth")
CURVES_PLOT_PATH = os.path.join(MODELS_DIR, "training_curves.png")

os.makedirs(MODELS_DIR, exist_ok=True)

def train_model(epochs=35, batch_size=16, lr=1e-3, val_split=0.2):
    if not os.path.exists(MANIFEST_PATH):
        print(f"[Error] Dataset manifest not found at: {MANIFEST_PATH}")
        print("Run 'python data_collector.py' or 'python ingest_archival_videos.py' first to collect data!")
        return
        
    print(f"Loading dataset from: {MANIFEST_PATH}")
    train_loader, val_loader = get_dataloaders(MANIFEST_PATH, batch_size=batch_size, val_ratio=val_split)
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = SpatioTemporalWaveNet().to(device)
    
    mse_criterion = nn.MSELoss()
    ce_criterion = nn.CrossEntropyLoss()
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    history = {
        "train_loss": [], "val_loss": [],
        "val_mse": [], "val_acc": []
    }
    
    best_val_loss = float("inf")
    
    print("\n" + "="*60)
    print("--- STARTING SPATIO-TEMPORAL WAVENET TRAINING ---")
    print("="*60)
    
    for epoch in range(1, epochs + 1):
        # 1. Training Phase
        model.train()
        train_loss_accum = 0.0
        
        for batch_x, target_q, target_r, _ in train_loader:
            batch_x = batch_x.to(device)
            target_q = target_q.to(device)
            target_r = target_r.to(device)
            
            optimizer.zero_grad()
            pred_q, pred_r = model(batch_x)
            
            loss_q = mse_criterion(pred_q, target_q)
            loss_r = ce_criterion(pred_r, target_r)
            total_loss = loss_q + 0.5 * loss_r
            
            total_loss.backward()
            optimizer.step()
            
            train_loss_accum += total_loss.item() * len(batch_x)
            
        scheduler.step()
        train_loss = train_loss_accum / len(train_loader.dataset)
        
        # 2. Validation Phase
        model.eval()
        val_loss_accum = 0.0
        val_mse_accum = 0.0
        correct_regime = 0
        total_val = 0
        
        with torch.no_grad():
            for batch_x, target_q, target_r, _ in val_loader:
                batch_x = batch_x.to(device)
                target_q = target_q.to(device)
                target_r = target_r.to(device)
                
                pred_q, pred_r = model(batch_x)
                
                loss_q = mse_criterion(pred_q, target_q)
                loss_r = ce_criterion(pred_r, target_r)
                total_loss = loss_q + 0.5 * loss_r
                
                val_loss_accum += total_loss.item() * len(batch_x)
                val_mse_accum += loss_q.item() * len(batch_x)
                
                preds_class = torch.argmax(pred_r, dim=1)
                correct_regime += (preds_class == target_r).sum().item()
                total_val += len(batch_x)
                
        val_loss = val_loss_accum / total_val if total_val > 0 else 0.0
        val_mse = val_mse_accum / total_val if total_val > 0 else 0.0
        val_acc = (correct_regime / total_val * 100.0) if total_val > 0 else 0.0
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mse"].append(val_mse)
        history["val_acc"].append(val_acc)
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "val_mse": val_mse,
                "val_acc": val_acc
            }, BEST_MODEL_PATH)
            save_star = " [BEST MODEL SAVED]"
        else:
            save_star = ""
            
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Val MSE: {val_mse:.4f} | Regime Acc: {val_acc:.1f}%{save_star}")
        
    print("\n" + "="*60)
    print(f"Training Complete! Best model saved to: {BEST_MODEL_PATH}")
    print(f"Final Best Val Loss: {best_val_loss:.4f}")
    
    # Plot Training Curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Multi-Task Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history["val_acc"], label="Regime Accuracy (%)", color="green")
    plt.title("Validation Regime Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(CURVES_PLOT_PATH)
    print(f"Saved training curve plot to: {CURVES_PLOT_PATH}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    
    train_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
