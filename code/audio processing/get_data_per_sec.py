import os
import io
import math
import wave
import torch
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from pyannote.audio import Pipeline
from huggingface_hub import login
# from setting.account_setting import huggingface_use_auth_token

def main(raw_audio_folder, csv_output_folder, output_audio_seg_folder, model_name, huggingface_use_auth_token):
    # 確保設備可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 登錄 Hugging Face 並初始化 Pipeline
    login(huggingface_use_auth_token)
    pipeline = Pipeline.from_pretrained(model_name, use_auth_token=huggingface_use_auth_token).to(device)

    # 處理音頻檔案
    for audio_file in tqdm(os.listdir(raw_audio_folder), desc="Processing audios", unit="file"):
        if not audio_file.endswith('vocals.wav'):
            continue
        if audio_file[:-11]+".csv" in os.listdir(csv_output_folder):
            print(f"Csv file of {audio_file} exist.")
            continue

        audio_path = os.path.join(raw_audio_folder, audio_file)
        print(f"Processing {audio_file}...")

        # 讀取音訊數據
        audio_tensor, sample_rate, total_duration = extract_audio_to_tensor(audio_path)
        audio_tensor = audio_tensor.to(device)

        # 使用 pyannote Pipeline
        diarization = pipeline({"waveform": audio_tensor, "sample_rate": sample_rate})
        seg_folder = os.path.join(output_audio_seg_folder,audio_file[:-11])
        header, output_segments = segment_analysis(audio_path,sample_rate,seg_folder,diarization, total_duration)

        # 保存結果
        output_file = os.path.join(csv_output_folder, f"{audio_file[:-11]}.csv")
        df = pd.DataFrame(output_segments, columns = header)
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

def extract_audio_to_tensor(audio_path):
    """直接讀取 WAV 音頻為 PyTorch Tensor"""
    with wave.open(audio_path, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        n_channels = wav_file.getnchannels()
        n_frames = wav_file.getnframes()  # 總樣本數

        audio = np.frombuffer(wav_file.readframes(n_frames), dtype=np.int16)

        # 計算音訊的總長（以秒為單位）
        total_duration = n_frames / sample_rate

    # if n_channels > 1:
    #     audio = audio[::n_channels]  # 確保單聲道輸出

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels)  # 重塑為 (樣本數, 聲道數)
        audio = audio.mean(axis=1)  # 每個時間點取平均

    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0) / 32768.0  # 標準化至 [-1, 1]
    return audio_tensor, sample_rate, total_duration

def save_segment(audio_path, start_time, end_time, segment_path, sample_rate):
    """儲存音訊片段"""
    with wave.open(audio_path, "rb") as wav_file:
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        start_frame = math.floor(start_time * sample_rate)
        end_frame = math.ceil(end_time * sample_rate)

        wav_file.setpos(start_frame)
        segment_data = wav_file.readframes(end_frame - start_frame)

    os.makedirs(os.path.dirname(segment_path), exist_ok=True)
    with wave.open(segment_path, "wb") as out_file:
        out_file.setnchannels(n_channels)
        out_file.setsampwidth(sample_width)
        out_file.setframerate(sample_rate)
        out_file.writeframes(segment_data)

def segment_analysis(audio_path,sample_rate,seg_folder, diarization,total_duration):
    # 步驟 1：統計每位說話者的總說話時間並初始化每秒狀態
    speaker_durations = {}
    speaker_count = 0  # 記錄說話者數量
    max_time = total_duration # 最大時間，用於決定最大時間長度

    for segment, track, speaker in diarization.itertracks(yield_label=True):
        print(f"{segment.start:.3f} --> {segment.end:.3f} {speaker}")
        duration = segment.end - segment.start

        # 更新最大時間
        max_time = max(max_time, segment.end)
        
        # 統計每位說話者的時長
        if speaker not in speaker_durations:
            speaker_durations[speaker] = []
            speaker_count += 1  # 新增說話者時，增加說話者數量
        '''
        注意，在1秒內的說話者聲音會在此步驟被過濾
        '''
        if duration <= 1.0:
            continue

        speaker_folder = os.path.join(seg_folder, speaker)
        segment_filename = f"{math.floor(segment.start)}_{math.ceil(segment.end)}.wav"
        segment_path = os.path.join(speaker_folder, segment_filename)
        save_segment(audio_path, segment.start, segment.end, segment_path, sample_rate)





        # 將每位說話者的發言時段以每秒為單位加入
        for second in range(math.floor(segment.start), math.ceil(segment.end)):
            speaker_durations[speaker].append(second)

    # 印出說話者人數
    print(f"總共有 {speaker_count} 位說話者。")

    # 步驟 2：每秒統計說話者發言情況
    speaker_seconds = {speaker: [False] * math.ceil(max_time) for speaker in speaker_durations}

    # 填充每秒的發言狀態
    for speaker, times in speaker_durations.items():
        for second in times:
            speaker_seconds[speaker][second] = True

    # 步驟 3：準備輸出每秒的統計結果
    output_segments = []
    header = ["second"] + ["total_num"] + list(speaker_durations.keys())  # 輸出標題行

    # 生成每秒鐘的發言狀況
    for second in range(math.ceil(max_time)):
        row = []  # 以 1 秒為起始時間
        total_num = 0  # 當前秒鐘發言的總人數
        
        # 記錄每位說話者的發言狀況並計算發言的總人數
        for speaker in speaker_durations.keys():
            is_speaking = speaker_seconds[speaker][second]
            row.append(is_speaking)
            if is_speaking:
                total_num += 1
        
        # 加入總發言人數
        row = [second + 1] + [total_num] + row
        output_segments.append(row)

    # 輸出結果
    print(f"每秒的說話者統計:")
    print(",".join(header))  # 輸出標題
    for row in output_segments:
        print(",".join(map(str, row)))  # 輸出每秒狀況

    return header,output_segments    

if __name__ == "__main__":

    # Set HF_TOKEN as an environment variable (e.g. `export HF_TOKEN=hf_xxx`),
    # or a local .env file loaded via python-dotenv. Never commit real tokens.
    huggingface_use_auth_token = os.environ["HF_TOKEN"]
    raw_audio_folder = os.environ.get("RAW_AUDIO_FOLDER", "")
    output_audio_seg_folder = os.environ.get("OUTPUT_AUDIO_SEG_FOLDER", "")
    csv_output_folder = os.environ.get("CSV_OUTPUT_FOLDER", "")
    model_name = "pyannote/speaker-diarization-3.1"
    main(raw_audio_folder,csv_output_folder,output_audio_seg_folder,model_name,huggingface_use_auth_token)