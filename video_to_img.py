import os
import cv2
import pandas as pd
from tqdm import tqdm

# === 設定 ===
csv_path = 'F:/APTOS/aptos_ophnet_new2/APTOS_train-val_annotation.csv'  # アノテーションCSV
video_root = 'F:/APTOS/aptos_ophnet_new2/aptos_videos'                  # 動画フォルダ
output_root = 'F:/output_images_phase_classification'                   # 画像出力先
save_fps = 1  # 1秒間隔でフレーム抽出

# === 出力先フォルダの作成 ===
os.makedirs(output_root, exist_ok=True)
df = pd.read_csv(csv_path)
print(f"Loaded annotation CSV. Total records: {len(df)}")

# === アノテーションの各行に対して処理 ===
for idx, row in tqdm(df.iterrows(), total=len(df)):
    video_id = row['video_id']
    start = float(row['start'])
    end = float(row['end'])
    split = row['split']
    phase_id = str(row['phase_id'])  # phase_id はフォルダ名として使用

    video_path = os.path.join(video_root, f"{video_id}.mp4")
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        continue

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # end値が動画時間を超えていないか確認し、必要なら補正
    if end > duration:
        if end - duration > 0.1:
            print(f"End time exceeds video length. Adjusted: {end:.2f} -> {duration:.2f}")
        end = duration

    # 抽出対象のフレーム範囲を設定
    start_frame = int(start * fps)
    end_frame = int(end * fps)
    step = max(1, int(fps / save_fps))  # 1秒間隔

    # phase_idごとに保存先フォルダを作成
    save_dir = os.path.join(output_root, split, phase_id)
    os.makedirs(save_dir, exist_ok=True)

    # フレームを抽出して保存
    for frame_idx in range(start_frame, end_frame + 1, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame at index {frame_idx}")
            break

        timestamp_sec = int(frame_idx / fps)
        filename = f"{video_id}_t{timestamp_sec:04d}.jpg"
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, frame)

    cap.release()

# === 完了メッセージ ===
print("All frame extraction and saving completed.")
