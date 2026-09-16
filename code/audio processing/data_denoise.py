import os
import subprocess
from demucs.separate import main as demucs_separate

def process_video(input_folder, output_folder, SHIFTS):
    os.makedirs(output_folder, exist_ok=True)
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".mp4"):
                input_video_path = os.path.join(root,file)

                extracted_audio_path = os.path.join(output_folder, os.path.splitext(file)[0] + "_audio.wav")
                output_vocals_path = os.path.join(output_folder, os.path.splitext(file)[0] + "_vocals.wav")

                print(f"處理影片檔案: {file}")

                # 使用 FFmpeg 提取音訊
                try:
                    subprocess.run(
                        [
                            "ffmpeg", 
                            "-i", input_video_path,
                            "-vn",  # 無視視訊
                            "-acodec", "pcm_s16le",  # 使用無損編碼
                            "-ar", "44100",  # 取樣率 44.1 kHz
                            "-ac", "2",  # 立體聲
                            extracted_audio_path
                        ],
                        check=True
                    )
                    print(f"音訊提取完成: {extracted_audio_path}")
                except subprocess.CalledProcessError as e:
                    print(f"FFmpeg 提取失敗: {e}")
                    continue

                # 使用 Demucs 提取人聲
                try:
                    demucs_separate([
                        "--two-stems", "vocals",  # 僅提取人聲
                        "-n", "mdx_extra",  # 使用模型名稱
                        "--shifts", str(SHIFTS),  # 時域增強次數
                        extracted_audio_path
                    ])
                    # 提取的文件會被保存到 Demucs 預設輸出目錄，移動到指定位置
                    demucs_output_dir = os.path.join("separated", "mdx_extra")
                    vocals_path = os.path.join(demucs_output_dir, os.path.splitext(os.path.basename(extracted_audio_path))[0], "vocals.wav")
                    os.rename(vocals_path, output_vocals_path)
                    print(f"人聲提取完成: {output_vocals_path}")

                except Exception as e:
                    print(f"Demucs 處理失敗: {e}")
                    continue

                # 清理暫存音訊
                os.remove(extracted_audio_path)
                print(f"處理完成: {output_vocals_path}")

if __name__ == "__main__":
    SHIFTS = 4  # 時域增強次數 :
    input_folder = os.environ.get("INPUT_VIDEO_FOLDER", "")
    output_folder = os.environ.get("OUTPUT_AUDIO_FOLDER", "")
    process_video(input_folder, output_folder, SHIFTS)