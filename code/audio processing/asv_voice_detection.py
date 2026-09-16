"""
Automatic speaker verification (ASV): matches each detected-speaker audio
segment (from get_data_per_sec.py's diarization output) against reference
recordings of the target candidate, using cosine similarity of ReDimNet
speaker embeddings. Corresponds to the "Speaker Recognition" step in the
paper's Online Appendix D (Yakovlev et al. 2024).

Model: loaded on demand via `torch.hub.load('IDRnD/ReDimNet', ...)` from
https://github.com/IDRnD/ReDimNet -- no local model weights need to be
vendored in this repository; torch downloads and caches the model
automatically the first time this script runs (requires internet access
and a Hugging Face / torch hub cache directory).
"""
import os
import librosa
import torch
import pandas as pd
from torch.nn.functional import cosine_similarity

def main(test_folder_path,candidate_folder_path,csv_file_out_path,device):
    model = get_model(device)
    candicate_list = get_candicate_list(candidate_folder_path)
    candidate_embeddings = get_candidate_embedding(model, candicate_list, candidate_folder_path,device)
    # print(f"Embedding set : {candidate_embeddings}")
    cant_identify = 0
    all_files_count = 0
    test_set = []
    for dirs in os.listdir(test_folder_path):
        name = "_".join(dirs[:-12].split("_")[3:])
        speakers = {}
        try :
            candidate_embeddings_avg = candidate_embeddings[name]
        except:
            print(f"{name} is a new candicate")
            continue
        for speaker_folder in os.listdir(os.path.join(test_folder_path,dirs)):
            for seg_file in os.listdir(os.path.join(test_folder_path,dirs,speaker_folder)):
                test_set = []
                seg_file_path = os.path.join(test_folder_path,dirs,speaker_folder,seg_file)
                embedding = get_embedding(seg_file_path,model,device)
                test_set.append(embedding)
            audio_set_set_tensor = torch.stack(test_set)
            test_embeddings = torch.mean(audio_set_set_tensor, dim=0).to('cpu')
            score = cal_cosine_similarity(test_embeddings,candidate_embeddings_avg)
            speakers[speaker_folder] = score
        torch.cuda.empty_cache()
        all_files_count +=1
        # 初始化變數
        highest_score = float("-inf")
        best_speaker = None
        # 遍歷字典，找到相合度最高的 speaker
        for speaker, score in speakers.items():
            # print(f"Each : {speaker,score}")
            if score > highest_score:
                highest_score = score
                best_speaker = speaker
        if (highest_score > 0.5 and highest_score < 0.7):
            cant_identify +=1
            # print(f"Total : {dirs,best_speaker,highest_score}")

        try :
            new_row = pd.DataFrame({"source": [name.replace("_"," ")],
                                    "post_id": [dirs[-11:]],
                                    "most_similar_speaker" : [best_speaker],
                                    "score" : [highest_score]
                                    })
            out_df = pd.concat([out_df, new_row], ignore_index=True)
        except :
            out_df = pd.DataFrame({"source": [name],
                                    "post_id": [dirs[-11:]],
                                    "most_similar_speaker" : [best_speaker],
                                    "score" : [highest_score]
                                    })
    out_df.to_excel(csv_file_out_path)
    print(all_files_count)


def get_model(device):
    model_name='M' # ~b3-b4 size
    train_type='ft_mix'
    dataset='vb2+vox2+cnc'
    model = torch.hub.load('IDRnD/ReDimNet', 'ReDimNet',
                        model_name=model_name,
                        train_type=train_type,
                        dataset=dataset)
    model.eval()
    return model

def get_candicate_list(candidate_folder_path):
    candicate_list = os.listdir(candidate_folder_path)
    return candicate_list

def get_candidate_embedding(model, candicate_list, candidate_folder_path,device):
    candidate_embeddings = {}
    for name in candicate_list:
        candicate_folder = os.path.join(candidate_folder_path,name)
        candidate_set = []
        for root, dir , seg_files in os.walk(candicate_folder):
            for seg_file in seg_files:
                if seg_file.endswith(".wav"):
                    seg_file_path = os.path.join(root,seg_file)
                    embedding = get_embedding(seg_file_path,model,device)
                    candidate_set.append(embedding)
        audio_set_set_tensor = torch.stack(candidate_set)
        candidate_embeddings[name] = torch.mean(audio_set_set_tensor, dim=0).to('cpu')

    return candidate_embeddings

# 處理音訊並計算相似度
def get_embedding(seg_file_path,model,device):
    model.to(device)
    audio_to_embedding, sample_rate = librosa.load(seg_file_path, sr=16000)
    audio_to_embedding = torch.tensor(audio_to_embedding).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(audio_to_embedding)
    return embedding

# cal_cosine_similarity of two list of tensor
def cal_cosine_similarity(test_embeddings,candidate_embeddings_avg):
    similarity = cosine_similarity(test_embeddings,candidate_embeddings_avg)
    # print(f"Cosine Similarity: {similarity}")
    return similarity.tolist()[0]

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # test_folder_path: per-video/per-speaker segmented audio, as produced by
    # get_data_per_sec.py's diarization step.
    test_folder_path = os.environ.get("TEST_SEGMENTS_FOLDER", "./data/ig_audios_segments")
    # candidate_folder_path: one subfolder per candidate, each containing
    # reference .wav recordings of that candidate's voice.
    candidate_folder_path = os.environ.get("CANDIDATE_VOICES_FOLDER", "./data/candidate_voices")
    csv_file_out_path = os.environ.get("OUTPUT_XLSX", "./output/speaker_detection.xlsx")
    main(test_folder_path,candidate_folder_path,csv_file_out_path,device)
