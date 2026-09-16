import os
import torch
import torch.nn as nn
import librosa
import statistics
import numpy as np
import pandas as pd
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

# 使用了預訓練的模型wav2vec2-large-robust-12-ft-emotion-msp-dim，因此架構不能任意更動
# 關於模型細節，請參考https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
# 以及論文 : Dawn of the transformer era in speech emotion recognition: closing the valence gap

# 接在預訓練模型最後方的線性回歸層，輸出情緒分數
class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    
# 這是尚未載入預訓練參數的模型架構 : 包含上游的Wav2Vec2Model，以及下游的線性回歸層
class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        # 將輸入通過
        outputs = self.wav2vec2(input_values)
        # 取最後一層的輸出
        hidden_states = outputs[0]
        # 作平均池化
        hidden_states = torch.mean(hidden_states, dim=1)
        # 進入線性回歸層
        logits = self.classifier(hidden_states)
        return hidden_states, logits
    
# 處理單一音訊檔案，回傳情緒分數
def process_audio(device, processor,model,file_path, sampling_rate=16000):
    try:
        # 讀取音訊
        signal, sr = librosa.load(file_path, sr=sampling_rate)
        signal = signal.reshape(1, -1)
        # 進行預處理，並將預處理後的音訊移動到GPU
        inputs = processor(signal, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        inputs = inputs.input_values.to(device)
        # 獲取情緒輸出，因為沒有要進行微調，運算過程中不紀錄梯度更新的資訊，可以大幅加快計算速度
        with torch.no_grad():
            _, logits = model(inputs)

        # 將輸出移回cpu，並回傳情緒分數 (Arousal, Dominance, Valence)
        return logits.cpu().numpy()[0]
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def add_names_cols(folder_name,avg_emotions,std_emotions):
    """ 添加檔案與資料夾相關資訊到 DataFrame """
    # 從 folder_name 提取日期、名字、短碼
    parts = folder_name.split("_")
    if len(parts) >= 5:
        date_part = parts[0]  # "2024-09-14"
        name = parts[3] + " " + parts[4]  # "Kari Lake"
        short_code = folder_name[-11:]  # "4xZsRKiYY"
        arousal_mean,dominance_mean,valence_mean = avg_emotions
        arousal_std,dominance_std,valence_std = std_emotions
    else:
        date_part, name, short_code = "", "", ""

    return pd.DataFrame([[folder_name, date_part, name, short_code,arousal_mean,
                          dominance_mean,valence_mean,
                          arousal_std,dominance_std,valence_std]], 
                        columns=["file_name", "date", "name", "short_code",
                                "arousal_mean", "dominance_mean", "valence_mean",
                                "arousal_std","dominance_std","valence_std"])

def main(device, processor,model,root_folder_path,output_csv_path):
    df_list = []
    for folder_name in os.listdir(root_folder_path):
        folder_path = os.path.join(root_folder_path, folder_name)
        if not os.path.isdir(folder_path):  # 確保是資料夾
            continue
        
        avg_emotion_list = []  # 用來存儲所有檔案的情緒資料
        for filename in os.listdir(folder_path):
            if filename.endswith(".wav"):
                try:
                    # 假設音訊檔案格式為 .wav
                    audio_file_path = os.path.join(folder_path, filename)
                    print(f"Now processing {audio_file_path}.")
                    # 獲取單一音訊的處理結果
                    emotions = process_audio(device, processor,model,audio_file_path)
                    if emotions is not None:
                        avg_emotion_list.append({
                            "arousal": emotions[0],
                            "dominance": emotions[1],
                            "valence": emotions[2],
                        })
                except Exception as e:
                    print(f"[錯誤] 無法處理 {audio_file_path}，錯誤: {e}")

        # 計算情緒特徵的平均值和標準差，如果資料夾(影片)只有一筆資料(句子)，標準差會補0，如果情緒資料缺失會補None
        if avg_emotion_list:
            arousal_avg = sum(e["arousal"] for e in avg_emotion_list) / len(avg_emotion_list)
            dominance_avg = sum(e["dominance"] for e in avg_emotion_list) / len(avg_emotion_list)
            valence_avg = sum(e["valence"] for e in avg_emotion_list) / len(avg_emotion_list)

            arousal_std = np.std([e["arousal"] for e in avg_emotion_list], ddof=1) if len(avg_emotion_list) > 1 else 0
            dominance_std = np.std([e["dominance"] for e in avg_emotion_list], ddof=1) if len(avg_emotion_list) > 1 else 0
            valence_std = np.std([e["valence"] for e in avg_emotion_list], ddof=1) if len(avg_emotion_list) > 1 else 0
        
        else:
            arousal_avg, dominance_avg, valence_avg = None, None, None
            arousal_std, dominance_std, valence_std = None, None, None
            

        df = add_names_cols(folder_name, [arousal_avg, dominance_avg, valence_avg], 
                            [arousal_std, dominance_std, valence_std])

        # 調整欄位順序，metadata 在前，特徵在後
        meta_columns = ["file_name", "date", "name", "short_code"]
        feature_columns = [col for col in df.columns if col not in meta_columns]  # 其他欄位 (特徵)
        df = df[meta_columns + feature_columns]  # 重新排列順序

        df_list.append(df)  
        print(f"Finished processing folder {folder_name}.")
    
    # 確保 df_list 不是空的，避免 ValueError
    if df_list:
        final_df = pd.concat(df_list, ignore_index=True)
        final_df.to_csv(output_csv_path, index=False)
        print(final_df.head())
    else:
        print("No valid audio files found. Skipping CSV creation.")
    final_df.to_csv(output_csv_path, index=False)

    



if __name__ == "__main__":
    # 設定模型與處理器
    # 如果可以的話使用GPU進行計算
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"DEVICE : {device}")
    # 設定預處理的流程並下載參數，最後使用to(device)將模型移動到GPU(如果有的話)
    model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = EmotionModel.from_pretrained(model_name).to(device)
    
    root_folder_path = os.environ.get("AUDIO_SENTENCE_FOLDER", "./data/ig_audios_sentence")
    output_csv_path = os.environ.get("OUTPUT_CSV", "./output/ig_audios_emotions.csv")
    main(device, processor,model,root_folder_path,output_csv_path)

