import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
from aptos_dataset import AptosIterableDataset
from Model.GL_MSTCN import GL_MSTCN
from utils import compute_duration_weights
from setting import *
import tqdm
import gc

def worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed + worker_id) 


def train_one_epoch(model, train_loader, criterion, optimizer, device, log_every=128):
    model.train()
    running_loss = 0.0
    running_samples = 0

    for inputs, labels, timestep, vid, frame_rate in tqdm.tqdm(train_loader):
        inputs = inputs.to(device)                # [B,3,224,224]
        labels = labels.clone().detach().view(-1).to(device)
        # labels = label_tensor.view(-1).to(device)       # [B]

        optimizer.zero_grad()
        x_mstcn = model(inputs, timestep, frame_rate)                 # logits: [B,35]
        loss = criterion(x_mstcn, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_samples += inputs.size(0)
        if running_samples % log_every == 0:
            avg = running_loss / running_samples
            print(f"\n[Train] processed {running_samples} samples, avg loss = {avg:.4f}")
        
        del x_mstcn
        torch.cuda.empty_cache()
        gc.collect()
    
    epoch_loss = running_loss / running_samples
    return epoch_loss


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels, timestep, vid, frame_rate in tqdm.tqdm(val_loader):
            # inputs = inputs.to(device)
            labels = labels.to(device)
            # labels = labels.view(-1).to(device)
            if isinstance(timestep, torch.Tensor) and timestep.numel() > 1:
                # 複数要素の場合 → ループで処理
                for inp, lbl, ts in zip(inputs, labels, timestep):
                    with torch.no_grad():
                        x_mstcn = model(inputs, timestep, frame_rate)
                    x_mstcn = model(inp.unsqueeze(0), ts, frame_rate)
                    loss = criterion(x_mstcn, lbl)

                    running_loss += loss.item() * inp.size(0)
                    total += inp.size(0)
                    preds = x_mstcn.argmax(dim=1)
                    correct += (preds == lbl).sum().item()
            else:
                with torch.no_grad():
                    x_mstcn = model(inputs, timestep, frame_rate)
                loss = criterion(x_mstcn, labels)

                running_loss += loss.item() * inputs.size(0)
                total += inputs.size(0)
                preds = x_mstcn.argmax(dim=1)
                correct += (preds == labels).sum().item()

    return running_loss / total, 100.0 * correct / total


# if __name__ == "__main__":
#     # ---- Config ----
#     VIDEO_DIR = "dataset/videos"
#     ANN_FILE  = "dataset/annotations/APTOS_train-val_annotation.csv"
#     SPLIT_TRAIN = "train"
#     SPLIT_VAL   = "val"
#     NUM_CLASSES = 35

#     BATCH_SIZE  = 32
#     NUM_WORKERS = 4
#     LR_BACKBONE = 5e-5
#     LR_HEAD     = 5e-4
#     NUM_EPOCHS  = 5
#     PATIENCE    = 3  # Number of epochs to wait for improvement before early stopping

#     # ---- Device ----
#     device = torch.device("cuda" if torch.cuda.is_available()
#                           else "mps"   if torch.backends.mps.is_available()
#                           else "cpu")
#     print("Using device:", device)

#     # ---- Datasets & Loaders ----
#     train_ds = AptosIterableDataset(
#         video_dir=VIDEO_DIR,
#         annotations_file=ANN_FILE,
#         split=SPLIT_TRAIN,
#         shuffle_videos=True,
#     )
#     val_ds = AptosIterableDataset(
#         video_dir=VIDEO_DIR,
#         annotations_file=ANN_FILE,
#         split=SPLIT_VAL,
#         shuffle_videos=True,
#     )

#     train_loader = DataLoader(
#         train_ds,
#         batch_size=None,
#         num_workers=NUM_WORKERS,
#         worker_init_fn=worker_init_fn
#     )
#     val_loader = DataLoader(
#         val_ds,
#         batch_size=None,
#         num_workers=NUM_WORKS_VAL,
#         worker_init_fn=worker_init_fn
#     )

#     # ---- Model ----
#     model = GL_MSTCN(out_channels_pretreatment=NUM_CLASSES*3*16, num_classes=NUM_CLASSES, basic_channels=BASIC_CHANNELS)
#     model.to(device)

#     for name, param in model.backbone.named_parameters():
#         if not name.startswith("layer4"):
#             param.requires_grad = False

#     for param in model.fc_phase.parameters():
#         param.requires_grad = True

#     for module in model.backbone.modules():
#         if isinstance(module, nn.BatchNorm2d):
#             module.eval()  

#     # ---- Loss & Optimizer ----
#     weights = compute_duration_weights(
#         annotations_file=ANN_FILE,
#         video_dir=VIDEO_DIR,
#         split=SPLIT_TRAIN,
#         num_classes=NUM_CLASSES
#     ).to(device)

#     criterion = nn.CrossEntropyLoss(weight=weights)

#     optimizer = optim.AdamW([
#         {"params": model.backbone.layer4.parameters(), "lr": LR_BACKBONE},
#         {"params": model.fc_phase.parameters(),           "lr": LR_HEAD},
#     ])


#     # ---- Training Loop ----
#     best_val_acc = 0.0
#     best_val_loss = float('inf')
#     patience_counter = 0
#     history = {"train_loss": [], "val_loss": [], "val_acc": []}

#     for epoch in range(1, NUM_EPOCHS + 1):
#         print(f"\n=== Epoch {epoch}/{NUM_EPOCHS} ===")
#         train_loss = train_one_epoch(model, train_loader,
#                                      criterion, optimizer, device)
#         val_loss, val_acc = validate(model, val_loader,
#                                      criterion, device)

#         history["train_loss"].append(train_loss)
#         history["val_loss"].append(val_loss)
#         history["val_acc"].append(val_acc)

#         print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | "
#               f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

#         # Early stopping check
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             patience_counter = 0
#             # Save best model
#             if val_acc > best_val_acc:
#                 best_val_acc = val_acc
#                 ckpt = "bresnet50_best.pth"
#                 torch.save(model.state_dict(), ckpt)
#                 print(f"  → New best model saved to {ckpt}")
#         else:
#             patience_counter += 1
#             print(f"  → No improvement in validation loss for {patience_counter} epochs")
#             if patience_counter >= PATIENCE:
#                 print(f"\nEarly stopping triggered after {epoch} epochs!")
#                 break

#     print(f"\nTraining finished. Best validation accuracy: {best_val_acc:.2f}%")

#     # ---- Plot Loss & Accuracy ----
#     import matplotlib.pyplot as plt
#     epochs = list(range(1, len(history["train_loss"]) + 1))
    
#     # Create plots directory if it doesn't exist
#     os.makedirs("plots", exist_ok=True)
    
#     plt.figure(figsize=(10,4))

#     plt.subplot(1,2,1)
#     plt.plot(epochs, history["train_loss"], label="Train")
#     plt.plot(epochs, history["val_loss"],   label="Val")
#     plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

#     plt.subplot(1,2,2)
#     plt.plot(epochs, history["val_acc"], label="Val Acc", color="green")
#     plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.legend()

#     plt.tight_layout()
    
#     # Save the plot
#     plot_path = os.path.join("plots", "training_history.png")
#     plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#     plt.close()
    
#     print(f"Training plots saved to {plot_path}")
