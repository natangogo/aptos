from aptos_dataset import AptosIterableDataset
from torch.utils.data import DataLoader
import torch
import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from setting import *
from Model.GL_MSTCN import GL_MSTCN

def test_process(model, pth_path_first, device):
    # データセットの準備（動画フレームの流れが得られる）
    test_dataset = AptosIterableDataset(PATH_APTOS_VIDEO, PATH_APTOS_CSV, split="val", shuffle_videos=False)
    test_loader = DataLoader(test_dataset, batch_size=None, num_workers=NUM_WORKS_TEST)

    # モデル読み込み
    model.load_state_dict(torch.load(pth_path_first))
    model.to(device)
    model.eval()

    # 推論用の辞書（動画単位で集める）
    video_dict = {}  # video_id -> {"frames": [], "labels": [], "timestamps": []}

    # バッチなしで1フレームずつ読み込む
    for frame, label, timestamp, video_id, frame_rate in tqdm.tqdm(test_loader):
        if video_id not in video_dict:
            video_dict[video_id] = {"frames": [], "labels": [], "timestamps": []}
        video_dict[video_id]["frames"].append(frame.squeeze(0))
        video_dict[video_id]["labels"].append(label)
        video_dict[video_id]["timestamps"].append(timestamp)
        video_dict[video_id]["frame_rate"] = frame_rate

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for video_id, data in video_dict.items():
            frames = torch.stack(data["frames"])  # [T, C, H, W]
            labels = torch.tensor(data["labels"])  # [T]
            timestamps = data["timestamps"]
            frame_rate = data["frame_rate"]

            if frames.dim() == 4:
                video = frames.unsqueeze(0).to(device)  # [1, T, C, H, W]
            else:
                video = frames.to(device)  # すでに [1, T, C, H, W] ならそのまま
            timestep = timestamps[0]

            # 推論（GL_MSTCNのforward）
            outputs = model(video, timestep, frame_rate)  # [T, num_classes]
            if outputs.dim() == 2:
                preds = torch.argmax(outputs, dim=1).cpu()
            elif outputs.dim() == 1:
                preds = torch.argmax(outputs).unsqueeze(0).cpu()
            else:
                raise ValueError(f"Unexpected output shape: {outputs.shape}")

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    # メトリクス計算
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)

    print("Accuracy:", acc)
    print("F1 Score (macro):", f1)
    print("Confusion Matrix:\n", cm)

    return acc, f1, cm
