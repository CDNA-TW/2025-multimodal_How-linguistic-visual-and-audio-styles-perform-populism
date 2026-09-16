import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import KFold
from datasets import Dataset

# Define label columns
label_cols = ['elite', 'centr', 'left', 'right']

# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

def load_and_format_multilabel_data(filepath, label_cols):
    df = pd.read_csv(filepath)
    aggregated = df.groupby('text_eng')[label_cols].any().astype(int).reset_index()
    return aggregated

def tokenize_and_tensorize(df, label_cols, tokenizer, max_length=512):
    encoded = tokenizer(
        list(df['text_eng']),
        truncation=True,
        padding=True,
        max_length=max_length
    )
    encoded["labels"] = df[label_cols].astype(np.float32).values.tolist()
    return Dataset.from_dict(encoded)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
        
    probs = 1 / (1 + np.exp(-logits))  # Sigmoid
    preds = (probs > 0.5).astype(int)

    if preds.shape != labels.shape:
        print(f"Shape mismatch: preds {preds.shape}, labels {labels.shape}")
        return {}
    
    return {
        "micro_f1": f1_score(labels, preds, average='micro', zero_division=0),
        "macro_f1": f1_score(labels, preds, average='macro', zero_division=0),
        "micro_precision": precision_score(labels, preds, average='micro', zero_division=0),
        "micro_recall": recall_score(labels, preds, average='micro', zero_division=0),
    }
    

def train_model(train_dataset, val_dataset, lr, batch_size, dir):
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-large",
        num_labels=4,
        problem_type="multi_label_classification" 
    )

    training_args = TrainingArguments(
        output_dir=f"{dir}/results",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=lr,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_steps=200,
        logging_steps=10,
        logging_dir=f"{dir}/logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return trainer

def grid_search_cv(df, learning_rates, batch_sizes, root_dir):
    kf = KFold(n_splits=5, shuffle=True, random_state=38)
    results = []

    for lr in learning_rates:
        for bs in batch_sizes:
            print(f"\nGrid search for LR={lr}, BS={bs}")
            fold = 0
            scores = []

            for train_idx, val_idx in kf.split(df):
                fold += 1
                train_df = df.iloc[train_idx]
                val_df = df.iloc[val_idx]

                print(f"Fold {fold}: train={len(train_df)}, val={len(val_df)}")
                
                train_dataset = tokenize_and_tensorize(train_df, label_cols, tokenizer)
                val_dataset = tokenize_and_tensorize(val_df, label_cols, tokenizer)

                trainer = train_model(train_dataset, val_dataset, lr, bs, f"{root_dir}/popbert38/fold{fold}-lr{lr}-bs{bs}")
                metrics = trainer.evaluate()
                print(f"Metrics returned from evaluation: {metrics}")
                
                if "eval_micro_f1" in metrics:
                    scores.append(metrics)
                else:
                    print("'eval_micro_f1' not found in metrics. Skipping this fold.")
                metrics_df = pd.DataFrame(scores)
                metrics_df.to_csv(f"{root_dir}/popbert38/fold{fold}-lr{lr}-bs{bs}/results/f1_scores_fold{fold}_{lr}_{bs}.csv", index=False)
            
            avg_micro_f1 = sum(d['eval_micro_f1'] for d in scores) / len(scores) if scores else 0.0
            avg_macro_f1 = sum(d['eval_macro_f1'] for d in scores) / len(scores) if scores else 0.0
            avg_micro_precision = sum(d['eval_micro_precision'] for d in scores) / len(scores) if scores else 0.0
            avg_micro_recall = sum(d['eval_micro_recall'] for d in scores) / len(scores) if scores else 0.0
            
            results.append({'lr': lr, 'batch_size': bs, 'avg_micro_f1': avg_micro_f1, 'avg_macro_f1': avg_macro_f1, 
                            'avg_micro_precision': avg_micro_precision, 'avg_micro_recall': avg_micro_recall})
        results_df = pd.DataFrame(results)
    
    return results_df

def final_train_and_test(train_df, test_df, best_lr, best_bs, root_dir):
    train_dataset = tokenize_and_tensorize(train_df, label_cols, tokenizer)
    test_dataset = tokenize_and_tensorize(test_df, label_cols, tokenizer)

    trainer = train_model(train_dataset, test_dataset, best_lr, best_bs, f"{root_dir}/popbert38/popbert-final")
    final_metrics = trainer.evaluate()
    print("Final Test Performance:", final_metrics)
    return trainer, final_metrics

# Set ROOT_DIR to the folder containing data/train_set.csv and data/test_set.csv
root_dir = os.environ.get("ROOT_DIR", ".")
learning_rates = [2e-5, 1.75e-5, 1.5e-5]
batch_sizes = [8]

# Load data
train_df = load_and_format_multilabel_data(f"{root_dir}/data/train_set.csv", label_cols)
test_df = load_and_format_multilabel_data(f"{root_dir}/data/test_set.csv", label_cols)

results_df = grid_search_cv(train_df, learning_rates, batch_sizes, root_dir)
results_df.to_csv(f"{root_dir}/popbert38/popbert_grid_results.csv", index=False)

# Select best
best = results_df.sort_values("avg_micro_f1", ascending=False).iloc[0]
best.to_csv(f"{root_dir}/popbert38/popbert_grid_results_best.csv", index=False)

trainer, final_metrics = final_train_and_test(train_df, test_df, best['lr'], int(best['batch_size']), root_dir)
trainer.save_model(f"{root_dir}/popbert38/popbert-final/model")
tokenizer.save_pretrained(f"{root_dir}/popbert38/popbert-final/tokenizer")

# Save final metrics
final_metrics_df = pd.DataFrame([final_metrics])
final_metrics_df.to_csv(f"{root_dir}/popbert38/popbert-final/final_metrics.csv", index=False)