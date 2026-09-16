# -*- coding: utf-8 -*-
"""
Applies a fine-tuned 7-class GoEmotion-style RoBERTa-large text-emotion
classifier to sentence-level transcripts, producing the per-sentence
anger_t/disgust_t/fear_t/joy_t/neutral_t/sadness_t/surprise_t probability
columns (plus Pred_t, the top predicted label) that
code/shared/feature_extraction.ipynb aggregates to per-video features.

Model weights are NOT included in this repository (the fine-tuned
checkpoint is ~1.4GB, too large for a normal git repo) -- set MODEL_DIR to
wherever you have placed emo_model_7class_v5/ (config.json +
model.safetensors), or host it externally (e.g. Hugging Face Hub, OSF) and
download it before running this script.
"""
import torch.nn.functional as F
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import numpy as np
import os
import pandas as pd

MODEL_DIR = os.environ.get("GOEMOTION_MODEL_DIR", "./model/emo_model_7class_v5")
MAX_LENGTH = 64
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

def preprocessing(input_text, tokenizer):
  '''
  Returns <class transformers.tokenization_utils_base.BatchEncoding> with the following fields:
    - input_ids: list of token ids
    - token_type_ids: list of token type ids
    - attention_mask: list of indices (0,1) specifying which tokens should considered by the model (return_attention_mask = True).
  '''
  return tokenizer.encode_plus(
                        input_text,
                        add_special_tokens=True,
                        padding='max_length',
                        truncation=True,
                        max_length=MAX_LENGTH,
                        return_tensors='pt'
                   )

model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR, output_attentions=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

emo_dict = {
    "0": 'anger',
    "1": 'disgust',
    "2": 'fear',
    "3": 'joy',
    "4": 'neutral',
    "5": 'sadness',
    "6": 'surprise'
}


def pred_sentence(text):
    encoding_dict = preprocessing(text, tokenizer)
    token_id = encoding_dict['input_ids'].to(device)
    attention_masks = encoding_dict['attention_mask'].to(device)
    output = model(token_id,
        token_type_ids = None,
        attention_mask = attention_masks)
    pred = np.argmax(output.logits.cpu().detach().numpy()).flatten().item()
    probabilities = F.softmax(output.logits.cpu(),dim=1).tolist()[0]
    return pred,probabilities


if __name__ == "__main__":
    # input_file: any CSV with a "sentence_text" column (sentence-level
    # transcript segments, e.g. from the audio-processing pipeline).
    input_file = os.environ.get("INPUT_CSV", "./data/sentence_transcripts.csv")
    output_file = os.environ.get("OUTPUT_CSV", "./output/labeled_text_emo.csv")

    df = pd.read_csv(input_file)
    texts = df["sentence_text"].astype(str).tolist()

    emo_names = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

    predictions = [pred_sentence(text) for text in texts]
    preds, probs = zip(*predictions) if predictions else ([], [])

    df['Pred_t'] = [emo_dict.get(str(p)) for p in preds]

    # Convert probability tuples to a 2D array for easier column assignment
    probs_array = np.array(probs)
    for i, emo in enumerate(emo_names):
        df[f'{emo}_t'] = probs_array[:, i]
    df.to_csv(output_file, index=False)
