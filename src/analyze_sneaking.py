import json
import pandas as pd
from pathlib import Path

# === Config ===
INPUT_PATH = Path("data/darkbench.jsonl")
OUTPUT_PATH = Path("outputs/model_responses.csv")
SNEAKING_PATTERN = "sneaking"

# === Load Data ===
def load_sneaking_prompts(filepath):
    prompts = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("metadata", {}).get("dark_pattern") == SNEAKING_PATTERN:
                prompts.append({
                    "id": obj["id"],
                    "prompt": obj["input"]
                })
    return pd.DataFrame(prompts)

# === Save Placeholder for Model Responses ===
def save_prompt_batch(df, path):
    df["response"] = ""  
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} prompts to {path}")

# === Main ===
if __name__ == "__main__":
    df = load_sneaking_prompts(INPUT_PATH)
    sneak_subset = df.head(20)  
    save_prompt_batch(sneak_subset, OUTPUT_PATH)
