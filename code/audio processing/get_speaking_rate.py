import os
import re
import pandas as pd
import whisper
from pydub import AudioSegment
import nltk
from nltk.corpus import cmudict
import numpy as np  # 用於處理 NaN

# 確保 NLTK 字典可用
nltk.download('cmudict')
cmu_dict = cmudict.dict()

def get_audio_length(audio_file: str) -> float:
    """ 使用 pydub 讀取音檔，回傳音檔長度（秒），若發生錯誤則回傳 0。 """
    try:
        audio = AudioSegment.from_file(audio_file)
        return audio.duration_seconds
    except Exception as e:
        print(f"[錯誤] 無法讀取音檔: {audio_file}，錯誤: {e}")
        return 0  # 讓後續檢查時能發現這是錯誤音檔

def whisper_process(audio_file: str, model) -> str:
    """ 使用 Whisper 轉錄音檔，回傳轉錄文字，若失敗則回傳空字串。 """
    try:
        result = model.transcribe(audio_file)
        return result["text"]
    except Exception as e:
        print(f"[錯誤] 轉錄失敗: {audio_file}，錯誤: {e}")
        return ""

def simple_fallback_syllable_count(word: str) -> int:
    """ 若 CMU 字典查不到單字，使用簡易母音群計算音節數。 """
    vowels = "aeiouy"
    count = 0
    prev_vowel = False

    for ch in word.lower():
        if ch in vowels:
            if not prev_vowel:
                count += 1
            prev_vowel = True
        else:
            prev_vowel = False

    return count if count > 0 else 1  # 確保至少有 1 個音節

def syllable_count(word: str) -> int:
    """ 
    嘗試使用 CMU 字典計算音節數，若查不到：
    1. 嘗試將首字母轉為大寫再查一次（可能是專有名詞）
    2. 仍然查不到時，使用簡易音節計算規則
    """
    word_lower = word.lower()
    word_capitalized = word.capitalize()

    # 先查小寫版
    if word_lower in cmu_dict:
        return sum(1 for p in cmu_dict[word_lower][0] if any(ch.isdigit() for ch in p))

    # 再查首字母大寫版（可能是專有名詞）
    if word_capitalized in cmu_dict:
        return sum(1 for p in cmu_dict[word_capitalized][0] if any(ch.isdigit() for ch in p))

    # 最後使用 fallback 方法
    return simple_fallback_syllable_count(word)

def nltk_process(text: str) -> int:
    """ 計算整段文字的總音節數 """
    words = re.sub(r'[^a-zA-Z\s]', '', text).split()
    return sum(syllable_count(word) for word in words)

def process_audio_files(data_folder: str, output_csv: str):
    """
    遍歷 `data_folder` 內的所有子資料夾，處理 `.wav` 檔案並計算語速，存入 CSV。
    """

    model = whisper.load_model("large")  # 只載入一次，提高效率
    results = []

    for folder_name in os.listdir(data_folder):
        # folder_name "2024-09-14_06-12-58_UTC_Kari_Lake_C_4xZsRKiYY"
        folder_path = os.path.join(data_folder, folder_name)
        
        # 檢查是否為資料夾
        if not os.path.isdir(folder_path):
            continue
        
        # 解析檔案名稱資訊
        try:
            date_part = folder_name.split("_")[0]  # 取得日期 (2024-09-01)
            name =  folder_name.split("_")[3] + " " +  folder_name.split("_")[4]# 固定名稱
            short_code = folder_name[-11:]  # 
        except IndexError:
            print(f"[警告] 無法解析資料夾名稱: {folder_name}")
            continue

        for file in os.listdir(folder_path):
            if file.endswith(".wav"):
                audio_file_path = os.path.join(folder_path, file)

                # 取得音檔長度
                audio_length = get_audio_length(audio_file_path)

                if audio_length == 0:  
                    print(f"[錯誤] 無效音檔: {audio_file_path}，將填入空值")
                    results.append({
                        "file_name": folder_name,
                        "segment_name" : file,
                        "date": date_part,
                        "name": name,
                        "short_code": short_code,
                        "speaking_rate": np.nan  # 保持 NaN 值
                    })
                    continue
                
                # Whisper 轉錄
                transcript = whisper_process(audio_file_path, model)

                if not transcript:  # 轉錄失敗也填 NaN
                    print(f"[錯誤] 無效音檔: {audio_file_path}，將填入空值")
                    results.append({
                        "file_name": folder_name,
                        "segment_name" : file,
                        "date": date_part,
                        "name": name,
                        "short_code": short_code,
                        "speaking_rate": np.nan  # 保持 NaN 值
                    })
                    continue

                # 計算音節數
                total_syllables = nltk_process(transcript)

                # 計算語速 (音節數 / 秒)
                speaking_rate = total_syllables / audio_length if audio_length > 0 else np.nan

                # 儲存結果
                    
                results.append({
                    "file_name": folder_name,
                    "segment_name" : file,
                    "date": date_part,
                    "name": name,
                    "short_code": short_code,
                    "speaking_rate":  speaking_rate # 保持 NaN 值
                })

                print(f"[完成] {audio_file_path} 語速: {speaking_rate:.2f} 音節/秒")

    # 建立 DataFrame 並存入 CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"[INFO] 所有結果已儲存至 {output_csv}")

if __name__ == "__main__":
    data_folder = os.environ.get("AUDIO_SENTENCE_FOLDER", "./data/ig_audios_sentence")  # 設定你的音檔資料夾
    output_csv = "./speaking_rates.csv"  # 輸出 CSV 檔案名稱
    process_audio_files(data_folder, output_csv)

