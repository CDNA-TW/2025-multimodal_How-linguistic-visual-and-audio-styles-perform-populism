import os
import liwc
import pandas as pd
from collections import Counter

# LIWC2022_English.dic is a proprietary dictionary file (not included in this
# repo) that must be purchased/obtained separately from https://www.liwc.app/
LIWC_DIC_PATH = os.environ.get("LIWC_DIC_PATH", "LIWC2022_English.dic")
parse, category_names = liwc.load_token_parser(LIWC_DIC_PATH)

input_csv = os.environ.get(
    "INPUT_CSV",
    "../../dataset/combined_241001-241104_N398_caption_post_populism.csv",
)

df = pd.read_csv(input_csv, encoding="big5")
texts = df["text"].astype(str).tolist()

# One LIWC category-count row per document (not one pooled count over the
# whole corpus), so downstream analysis can join back to df by row index.
rows = []
for text in texts:
    tokens = text.split()
    rows.append(Counter(cat for token in tokens for cat in parse(token)))

features_df = pd.DataFrame(rows).fillna(0).astype(int)

output_csv = os.environ.get("OUTPUT_CSV", "liwc_features_output.csv")
features_df.to_csv(output_csv, index=False)
print(f"Saved: {output_csv}")
print(features_df.sum().sort_values(ascending=False))
