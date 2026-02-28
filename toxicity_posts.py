#!/usr/bin/env python3
import os
import gc
import pandas as pd
import torch
from detoxify import Detoxify
from tqdm import tqdm


# ================= CONFIG =================

INPUT_DIR = "/storage/subcollapse/raw_reddit_new/posts/"
OUTPUT_DIR = "/storage/subcollapse/features/antosocial_content/toxic_hate/posts/"

CHUNK_SIZE = 50_000      # safe for large files
BATCH_SIZE = 512         # detoxify inference batch

TEXT_COLS = ["title", "selftext"]

OUTPUT_COLS = [
    "id",
    "name",
    "subreddit",
    "author",
    "created",
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
]
# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Detoxify("original", device=device)

TOX_COLS = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
]

def infer(texts):
    with torch.no_grad():
        preds = model.predict(texts)
    return preds

def process_file(in_path, out_path):
    first_write = True

    for chunk in pd.read_csv(in_path, chunksize=CHUNK_SIZE):
        # Combine title + selftext
        text = (
            chunk.get("title", "").fillna("") + " " +
            chunk.get("selftext", "").fillna("")
        ).tolist()

        results = {k: [] for k in TOX_COLS}

        for i in range(0, len(text), BATCH_SIZE):
            batch = text[i:i + BATCH_SIZE]
            preds = infer(batch)
            for k in TOX_COLS:
                results[k].extend(preds[k])

        for k in TOX_COLS:
            chunk[k] = results[k]

        # Keep ONLY required columns (safe ordering)
        chunk = chunk.reindex(columns=OUTPUT_COLS)

        chunk.to_csv(
            out_path,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
        )
        first_write = False

        del chunk
        gc.collect()

def main():
    files = sorted(f for f in os.listdir(INPUT_DIR) if f.startswith("RS_") and f.endswith(".csv"))

    for fname in tqdm(files, desc="Processing RS files"):
        in_path = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(OUTPUT_DIR, fname)
        process_file(in_path, out_path)

if __name__ == "__main__":
    main()



