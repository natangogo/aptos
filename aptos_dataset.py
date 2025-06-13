import os
import glob
import csv
import random
from typing import Iterator, Tuple, Optional
from setting import *

# このコードは、APTOSというデータセットを使って、動画から1秒ごとに画像を抽出し、
# それに対するラベルを付けて、その画像とラベルをpytorchで使えるようにデータローダーとして返す仕組み

import torch
from torch.utils.data import IterableDataset, get_worker_info
# from torchcodec.decoders import VideoDecoder
import torchvision.transforms as T


def get_resnet50_transform() -> T.Compose:
    """Returns the standard ImageNet preprocessing pipeline for ResNet-50."""
    return T.Compose([
        T.ConvertImageDtype(torch.float),  # uint8 [0,255] -> float [0.0,1.0]
        T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(224),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

import os
import glob
import csv
import random
import numpy as np
from typing import Iterator, Tuple, Optional

import torch
from torch.utils.data import IterableDataset, get_worker_info
# from torchcodec.decoders import VideoDecoder
import torchvision.transforms as T
from decord import VideoReader
from decord import cpu
import pickle

def get_resnet50_transform() -> T.Compose:
    """Returns the standard ImageNet preprocessing pipeline for ResNet-50."""
    return T.Compose([
        T.ConvertImageDtype(torch.float),  # uint8 [0,255] -> float [0.0,1.0]
        T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(224),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


class AptosIterableDataset(IterableDataset):
    """
    IterableDataset that streams frames from videos at 1 frame per second.
    Yields (frame_tensor, label, timestamp).
    """
    def __init__(
        self,
        video_dir: str,
        annotations_file: str,
        split: str = 'train',
        transform: Optional[T.Compose] = None,
        shuffle_videos: bool = False,
    ):
        super().__init__()
        self.video_dir = video_dir
        self.transform = transform or get_resnet50_transform()
        self.split = split.lower()
        self.shuffle_videos = shuffle_videos

        # --- キャッシュファイルを定義 ---
        cache_file = f'cache_{self.split}.pkl'
        if os.path.exists(cache_file):
            # print(f"[INFO] Loading cached metadata from {cache_file}")
            with open(cache_file, 'rb') as f:
                self.video_files, self.annotations = pickle.load(f)
            return  # ← 残りの初期化はスキップ
        
        # Load annotations filtered by split
        self.annotations = {}  # vid -> [(start, end, phase_id), ...]
        with open(annotations_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get('split','train').lower() != self.split:
                    continue
                vid = row['video_id']
                start = float(row['start'])
                end   = float(row['end'])
                phase = int(row['phase_id'])
                self.annotations.setdefault(vid, []).append((start, end, phase))

        # Collect video file paths for this split
        all_videos = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
        self.video_files = []
        for path in all_videos:
            vid = os.path.splitext(os.path.basename(path))[0]
            if vid in self.annotations:
                self.video_files.append(path)
        if not self.video_files:
            raise ValueError(f"No videos found for split='{self.split}' in {video_dir}")
        
        # --- キャッシュ保存 ---:キャッシュがある値、一瞬で復元で来ていちいち読み込無必要がなくなり時短に！
        # print(f"[INFO] Saving metadata cache to {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump((self.video_files, self.annotations), f)

    def _get_label_for_timestamp(self, timestamp: float, annotations: list) -> Optional[int]:
        timestamp = round(float(timestamp), 2)
        eps = 0.2  # 十分に広い誤差を許容

        for start, end, phase in annotations:
            if (start - eps) <= timestamp <= (end + eps):
                return phase
        return None

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, int, float]]:
        # Partition video list across workers
        worker_info = get_worker_info()
        if worker_info is None:
            vids = list(self.video_files)
        else:
            # Split video list evenly among workers
            vids = self.video_files[worker_info.id::worker_info.num_workers]

        # Optionally shuffle video order
        if self.shuffle_videos:
            random.shuffle(vids)

        # Stream frames from each assigned video
        for idx, path in enumerate(vids):
            # print("Processing video: ", path)
            try:
                print(f"[{idx+1}/{len(vids)}] Loading: {os.path.basename(path)}")
                vid = os.path.splitext(os.path.basename(path))[0]
                annots = self.annotations.get(vid, [])
                
                if not annots:  # Skip videos without annotations
                    continue

                try:
                    reader = VideoReader(path, ctx=cpu(0))
                    num_frames = len(reader)
                    frame_rate = reader.get_avg_fps()
                except Exception as e:
                    print(f"[Decord Error] Failed to open video {vid}: {e}")
                    continue
                duration = num_frames / frame_rate  # 秒数
                # print(f"{vid}: duration={duration:.2f}s, frame_rate={frame_rate:.2f}, num_frames={num_frames}")
                # === 対策 1: 不正な動画メタ情報をスキップ ===
                if duration <= 1 or frame_rate < 1:
                    print(f"[Warning] Skipping {vid}: invalid duration ({duration:.2f}s) or fps ({frame_rate:.2f})")
                    continue

                annots = [
                    (start, end, phase)
                    for (start, end, phase) in annots
                    if start < duration  # startが動画内にあれば有効とみなす
                ]
                if not annots:
                    print(f"Skipping {vid}: all annotations exceed duration ({duration:.2f}s)")
                    continue
                # 秒ごとに最も近いフレームを取得
                fps_step = FPS_STEP  # 0.25秒間隔 → 約4fps相当（任意で調整）

                second = 0.0
                frames = []
                timestamps = []

                while second < duration:
                    try:
                        frame_index = int(second * frame_rate)
                        if  frame_index >= num_frames - 1:
                            second += fps_step
                            continue
                        
                        timestamp_val = round(second, 2)

                        label = self._get_label_for_timestamp(timestamp_val, annots)
                        if label is None:
                            second += fps_step
                            continue

                        frame = reader[frame_index]
                        frame = torch.from_numpy(frame.asnumpy()).permute(2, 0, 1)  # [C, H, W]
                        frame = self.transform(frame)

                        frames.append(frame)
                        timestamps.append(timestamp_val)

                    except Exception as e:
                        print(f"[Frame Error] {vid}: frame_idx={frame_index} error: {e}")
                        second += fps_step
                        continue

                    second += fps_step  # ← ★これがwhileの最後に必須

                if len(frames) >= 8 and label is not None:  # 最低系列長を保証
                    frames_tensor = torch.stack(frames, dim=0)  # → [T, C, H, W]
                    final_label = timestamps and self._get_label_for_timestamp(timestamps[0], annots)
                    if final_label is not None:
                        yield frames_tensor, label, timestamps[0], vid, frame_rate
                    else:
                        print(f"[Skip] No valid label found in {vid}")
                else:
                    print(f"[Skip] Too few valid frames in {vid}")
                    continue  # 明示的にスキップ

            except Exception as e:
                # Skip problematic videos entirely
                # print(f"Failed to process video {path}: {e}")
                continue

# import pandas as pd

# ann_df = pd.read_csv(r"F:/APTOS/aptos_ophnet_new2/APTOS_train-val_annotation.csv")
# print("== Annotations for case_0012 ==")
# print(ann_df[ann_df["video_id"] == "case_0012"])

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    video_dir = PATH_APTOS_VIDEO
    ann_dir = PATH_APTOS_CSV
    # datasets and loaders
    train_ds = AptosIterableDataset(video_dir, ann_dir, split='train')
    train_loader = DataLoader(train_ds, batch_size=None)

    val_ds = AptosIterableDataset(video_dir, ann_dir, split='val')
    val_loader = DataLoader(val_ds, batch_size=None)

    for frames_batch, labels_batch, timestamps_batch, name, frame_rate in train_loader:
        # print(frames_batch.shape, labels_batch.shape, timestamps_batch.shape, f"label:{labels_batch}")
        print(f"video: {name}, timestamp: {timestamps_batch}, label: {labels_batch}")
