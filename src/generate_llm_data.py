import os
import time

import pandas as pd
import google.generativeai as genai
from anthropic import Anthropic
from openai import OpenAI

from prompts import PROMPTS


# ===============================
# CONFIG
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

INPUT_FILE = os.path.join(BASE_DIR, "data", "combined.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "Result", "LLM_Output_new")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use Gemini, ChatGPT, and Claude
LLMS = ["gemini", "chatgpt", "claude"]


# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(INPUT_FILE)

# IMPORTANT: keep small for testing
df = df.sample(n=1000, random_state=42)

texts = df["text"].tolist()

print("Total texts:", len(texts))


# ===============================
# GEMINI LLM FUNCTION
# ===============================
# PUT YOUR REAL KEY HERE
genai.configure(api_key="AIzaSyC1xZzbtjQJ_HvDKoPfdN_GHP3TNGmJ7wQ")

model_gemini = genai.GenerativeModel("gemini-pro")


# ===============================
# CHATGPT LLM FUNCTION
# ===============================
# PUT YOUR REAL OPENAI KEY HERE
openai_client = OpenAI(api_key="YOUR_OPENAI_KEY")
OPENAI_MODEL = "gpt-4o-mini"


# ===============================
# CLAUDE LLM FUNCTION
# ===============================
# PUT YOUR REAL ANTHROPIC KEY HERE
anthropic_client = Anthropic(api_key="YOUR_ANTHROPIC_KEY")
CLAUDE_MODEL = "claude-3-5-sonnet-latest"


def call_llm(text, prompt, model):
    full_prompt = prompt + "\n\n" + text

    try:
        if model == "gemini":
            response = model_gemini.generate_content(full_prompt)
            return response.text.strip()

        if model == "chatgpt":
            response = openai_client.responses.create(
                model=OPENAI_MODEL,
                input=full_prompt,
            )
            return response.output_text.strip()

        if model == "claude":
            response = anthropic_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1024,
                messages=[{"role": "user", "content": full_prompt}],
            )
            return response.content[0].text.strip()

        return text  # fallback (for now)

    except Exception as e:
        print("Error:", e)
        return text


# ===============================
# GENERATE DATASETS
# ===============================
for model in LLMS:
    for key, prompt in PROMPTS.items():
        print(f"\nRunning {model} + {key}")

        new_texts = []

        for i, t in enumerate(texts):
            new_text = call_llm(t, prompt, model)
            new_texts.append(new_text)

            if i % 20 == 0:
                print(f"{i}/{len(texts)} done")

            # IMPORTANT: helps with Gemini rate limits
            time.sleep(0.5)

        new_df = df.copy()
        new_df["text"] = new_texts

        output_path = os.path.join(OUTPUT_DIR, f"{model}_{key}.csv")
        new_df.to_csv(output_path, index=False)

        print("Saved:", output_path)


# ===============================
# SAVE ORIGINAL
# ===============================
df.to_csv(os.path.join(OUTPUT_DIR, "original.csv"), index=False)

print("\nAll datasets generated successfully!")
