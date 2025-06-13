import os
import shutil
from decord import VideoReader, cpu
import imageio_ffmpeg
import subprocess
import time

def is_broken_video(path):
    try:
        reader = VideoReader(path, ctx=cpu(0))
        if len(reader) == 0:
            print(f"⚠ フレーム数0: {os.path.basename(path)}")
            return True
        _ = reader[len(reader) // 2]  # 中央フレームをテスト読み込み
        return False
    except Exception as e:
        print(f"❌ 壊れている: {os.path.basename(path)}, 理由: {e}")
        return True

def repair_video(input_path, output_path):
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_path, "-y", "-i", input_path,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-c:a", "aac", "-strict", "experimental",
        output_path
    ]
    try:
        start = time.time()
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elapsed = time.time() - start
        print(f"🔧 修復保存: {os.path.basename(output_path)}（{elapsed:.1f}秒）")
    except subprocess.CalledProcessError:
        print(f"❌ 修復失敗: {os.path.basename(input_path)}")

def main():
    input_dir =  r"F:/APTOS/aptos_ophnet_new3/aptos_videos"             # 元の動画フォルダ
    output_dir =  r"F:/APTOS/aptos_ophnet_new3/fixed_videos"       # 修復後の保存先フォルダ
    os.makedirs(output_dir, exist_ok=True)

    video_files = [f for f in os.listdir(input_dir) if f.endswith(".mp4")]

    for video_file in sorted(video_files):
        input_path = os.path.join(input_dir, video_file)
        output_path = os.path.join(output_dir, video_file)

        if is_broken_video(input_path):
            repair_video(input_path, output_path)
        else:
            shutil.copy2(input_path, output_path)
            print(f"✅ 正常コピー: {video_file}")

if __name__ == "__main__":
    main()