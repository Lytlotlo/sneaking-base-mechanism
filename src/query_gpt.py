import pandas as pd
import openai
import time
from dotenv import load_dotenv
import os
from pathlib import Path

# === Load your API key ===
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# === File Paths ===
INPUT_PATH = Path("outputs/model_responses.csv")
OUTPUT_PATH = Path("outputs/model_responses_filled.csv")
LOG_PATH = Path("outputs/model_logs/gpt4_logs.txt")

# === Load Prompts ===
df = pd.read_csv(INPUT_PATH)
df["response"] = ""  

# === Create log folder if not exists ===
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# === Send to GPT-4 ===
with open(LOG_PATH, "w", encoding="utf-8") as log_file:
    for index, row in df.iterrows():
        prompt_text = f'Please respond naturally to the following instruction:\n\n"{row["prompt"]}"'
        print(f"→ Querying GPT-4: {row['id']}")

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0
            )
            reply = response["choices"][0]["message"]["content"].strip()
            df.at[index, "response"] = reply

            # Log for audit
            log_file.write(f"\n\nPROMPT ({row['id']}):\n{row['prompt']}\n")
            log_file.write(f"RESPONSE:\n{reply}\n")

            time.sleep(1.5)  # Avoid rate limit
        except Exception as e:
            print(f"Error at row {index}: {e}")
            df.at[index, "response"] = "ERROR"

# === Save Completed Output ===
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Saved completed responses to: {OUTPUT_PATH}")
