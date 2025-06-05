import os
import time
from setting import *
from Model.GL_MSTCN import GL_MSTCN
from train import train_one_epoch, validate, worker_init_fn
from test import test_process
from utils import compute_duration_weights
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from aptos_dataset import AptosIterableDataset

if __name__ == "__main__":
    date = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    print("Experiment Date:", date)

    save_root = os.path.join(SAVE_PATH, date)
    os.makedirs(save_root, exist_ok=True)
    save_train_path = os.path.join(save_root, "Train")
    save_test_path = os.path.join(save_root, "Test")
    os.makedirs(save_train_path, exist_ok=True)
    os.makedirs(save_test_path, exist_ok=True)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Dataset
    train_ds = AptosIterableDataset(PATH_APTOS_VIDEO, PATH_APTOS_CSV, split="train", shuffle_videos=True)
    val_ds = AptosIterableDataset(PATH_APTOS_VIDEO, PATH_APTOS_CSV, split="val", shuffle_videos=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_NUM_TRAIN, num_workers=NUM_WORKS_TRAIN, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_NUM_TRAIN, num_workers=NUM_WORKS_VAL, worker_init_fn=worker_init_fn)

    # Model
    model = GL_MSTCN(out_channels_pretreatment=NUM_CLASSES*3*8, num_classes=NUM_CLASSES_TRAIN, basic_channels=BASIC_CHANNELS)
    model.to(device)

    # Loss and optimizer
    weights = compute_duration_weights(PATH_APTOS_CSV, PATH_APTOS_VIDEO, split="train", num_classes=NUM_CLASSES_TRAIN).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    best_val_acc = 0.0
    best_model_path = os.path.join(save_train_path, "best_model.pth")

    print('Traing Start !!')
    for epoch in range(1, EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  → New best model saved to {best_model_path}")

    print("\nTraining completed. Best validation accuracy:", best_val_acc)

    # Evaluation
    print("\nStart Testing with Best Model...")
    acc, f1, cm = test_process(model, best_model_path, device)

    # Save results
    with open(os.path.join(save_test_path, "test_results.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm))

    print("Test results saved to:", save_test_path)
