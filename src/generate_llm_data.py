import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import joblib
import pandas as pd
from anthropic import Anthropic
from google import genai
from openai import OpenAI

from prompts import PROMPTS
from preprocess import clean_text


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai.models").setLevel(logging.WARNING)
logging.getLogger("google.genai").setLevel(logging.WARNING)


def load_local_env(env_path):
    if not os.path.exists(env_path):
        logger.warning("Local env file not found at %s", env_path)
        return

    logger.info("Loading environment variables from %s", env_path)

    with open(env_path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()

            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value


load_local_env(os.path.join(BASE_DIR, ".env"))


def get_env_or_default(name, default):
    value = os.getenv(name)
    return value if value else default

INPUT_FILE = os.path.join(BASE_DIR, "data", "combined.csv")
OUTPUT_DIR = get_env_or_default(
    "LLM_OUTPUT_DIR",
    os.path.join(BASE_DIR, "Result", "LLM_Output"),
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_PATH = os.path.join(BASE_DIR, "Result", "Model", "final_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "Result", "Model", "tfidf_vectorizer.pkl")

LLM_CONFIGS = {
    "gemini": {
        "api_key_env": "GEMINI_API_KEY",
        "model": get_env_or_default("GEMINI_MODEL", "gemini-2.5-flash"),
    },
    "chatgpt": {
        "api_key_env": "OPENAI_API_KEY",
        "model": get_env_or_default("OPENAI_MODEL", "gpt-4o-mini"),
    },
    "claude": {
        "api_key_env": "ANTHROPIC_API_KEY",
        "model": get_env_or_default("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
    },
}
ENABLED_LLM_NAMES = [
    llm_name.strip().lower()
    for llm_name in get_env_or_default("LLM_ENABLED_PROVIDERS", "gemini").split(",")
    if llm_name.strip()
]

REQUEST_DELAY_SECONDS = float(get_env_or_default("LLM_REQUEST_DELAY_SECONDS", "0.5"))
SAMPLE_SIZE = os.getenv("LLM_SAMPLE_SIZE")
MAX_RETRIES_PER_ROW = int(get_env_or_default("LLM_MAX_RETRIES_PER_ROW", "2"))
BATCH_SIZE = int(get_env_or_default("LLM_BATCH_SIZE", "5"))
MAX_WORKERS = int(get_env_or_default("LLM_MAX_WORKERS", str(BATCH_SIZE)))


class FatalProviderError(RuntimeError):
    pass


def extract_retry_delay_seconds(error_message):
    match = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", error_message, re.IGNORECASE)
    if not match:
        return None

    return max(1, int(float(match.group(1))) + 1)


def load_dataset():
    logger.info("Loading input dataset from %s", INPUT_FILE)
    df = pd.read_csv(INPUT_FILE)

    if SAMPLE_SIZE:
        sample_size = min(int(SAMPLE_SIZE), len(df))
        logger.info("Sampling %s rows from combined dataset", sample_size)
        df = df.sample(n=sample_size, random_state=42)

    logger.info("Dataset ready with %s texts", len(df))
    return df


def build_clients():
    clients = {}

    gemini_api_key = os.getenv(LLM_CONFIGS["gemini"]["api_key_env"])
    if gemini_api_key:
        clients["gemini"] = genai.Client(api_key=gemini_api_key)
        logger.info(
            "Gemini client initialized with model %s",
            LLM_CONFIGS["gemini"]["model"],
        )
    else:
        logger.warning("Gemini disabled: missing %s", LLM_CONFIGS["gemini"]["api_key_env"])

    openai_api_key = os.getenv(LLM_CONFIGS["chatgpt"]["api_key_env"])
    if openai_api_key:
        clients["chatgpt"] = OpenAI(api_key=openai_api_key)
        logger.info(
            "ChatGPT client initialized with model %s",
            LLM_CONFIGS["chatgpt"]["model"],
        )
    else:
        logger.warning("ChatGPT disabled: missing %s", LLM_CONFIGS["chatgpt"]["api_key_env"])

    anthropic_api_key = os.getenv(LLM_CONFIGS["claude"]["api_key_env"])
    if anthropic_api_key:
        clients["claude"] = Anthropic(api_key=anthropic_api_key)
        logger.info(
            "Claude client initialized with model %s",
            LLM_CONFIGS["claude"]["model"],
        )
    else:
        logger.warning("Claude disabled: missing %s", LLM_CONFIGS["claude"]["api_key_env"])

    return clients


def load_classifier_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise RuntimeError(
            "Missing classifier artifacts. Expected files at "
            f"{MODEL_PATH} and {VECTORIZER_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    logger.info("Loaded classifier artifacts for label-change tracking")
    return model, vectorizer


def get_enabled_llms(clients):
    enabled_llms = [
        llm_name
        for llm_name in LLM_CONFIGS
        if llm_name in clients and llm_name in ENABLED_LLM_NAMES
    ]

    if not enabled_llms:
        configured_llms = ", ".join(ENABLED_LLM_NAMES) if ENABLED_LLM_NAMES else "none"
        raise RuntimeError(
            "No enabled LLM clients are available. "
            f"Configured providers: {configured_llms}"
        )

    logger.info("Enabled LLMs: %s", ", ".join(enabled_llms))
    return enabled_llms


def call_llm(text, prompt, llm_name, clients, prompt_key, row_number):
    full_prompt = f"{prompt}\n\n{text}"

    for attempt in range(1, MAX_RETRIES_PER_ROW + 2):
        try:
            if llm_name == "gemini":
                response = clients["gemini"].models.generate_content(
                    model=LLM_CONFIGS["gemini"]["model"],
                    contents=full_prompt,
                )
                return response.text.strip()

            if llm_name == "chatgpt":
                response = clients["chatgpt"].responses.create(
                    model=LLM_CONFIGS["chatgpt"]["model"],
                    input=full_prompt,
                )
                return response.output_text.strip()

            if llm_name == "claude":
                response = clients["claude"].messages.create(
                    model=LLM_CONFIGS["claude"]["model"],
                    max_tokens=1024,
                    messages=[{"role": "user", "content": full_prompt}],
                )
                return "".join(
                    block.text for block in response.content if getattr(block, "text", None)
                ).strip()

            raise ValueError(f"Unsupported LLM: {llm_name}")

        except Exception as error:
            error_message = str(error)
            logger.error(
                "LLM call failed | llm=%s | prompt=%s | row=%s | attempt=%s | error=%s",
                llm_name,
                prompt_key,
                row_number,
                attempt,
                error_message,
            )

            if "reported as leaked" in error_message.lower() or "permission_denied" in error_message.lower():
                raise FatalProviderError(
                    f"{llm_name} rejected the API key. Update {LLM_CONFIGS[llm_name]['api_key_env']}."
                ) from error

            if "resource_exhausted" in error_message.lower() or "429" in error_message:
                retry_delay = extract_retry_delay_seconds(error_message)
                if retry_delay and attempt <= MAX_RETRIES_PER_ROW:
                    logger.warning(
                        "Rate limit hit | llm=%s | prompt=%s | row=%s | sleeping=%ss before retry",
                        llm_name,
                        prompt_key,
                        row_number,
                        retry_delay,
                    )
                    time.sleep(retry_delay)
                    continue

            if attempt <= MAX_RETRIES_PER_ROW:
                logger.warning(
                    "Retrying row after transient failure | llm=%s | prompt=%s | row=%s | next_attempt=%s",
                    llm_name,
                    prompt_key,
                    row_number,
                    attempt + 1,
                )
                time.sleep(REQUEST_DELAY_SECONDS)
                continue

            logger.error(
                "Falling back to original text | llm=%s | prompt=%s | row=%s",
                llm_name,
                prompt_key,
                row_number,
            )
            return text


def process_batch(batch_rows, prompt, llm_name, clients, prompt_key):
    batch_results = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_row_number = {
            executor.submit(
                call_llm,
                row["text"],
                prompt,
                llm_name,
                clients,
                prompt_key,
                row["row_number"],
            ): row["row_number"]
            for row in batch_rows
        }

        for future in as_completed(future_to_row_number):
            row_number = future_to_row_number[future]
            batch_results[row_number] = future.result()

    ordered_results = []
    fallback_count = 0

    for row in batch_rows:
        result_text = batch_results[row["row_number"]]
        ordered_results.append(result_text)

        if result_text == row["text"]:
            fallback_count += 1

    return ordered_results, fallback_count


def predict_labels(texts, model, vectorizer):
    cleaned_texts = [clean_text(text) for text in texts]
    features = vectorizer.transform(cleaned_texts)
    predictions = model.predict(features)
    return predictions.tolist()


def main():
    logger.info("Starting LLM dataset generation")
    logger.info("Output directory: %s", OUTPUT_DIR)
    logger.info("Request delay: %s seconds", REQUEST_DELAY_SECONDS)
    logger.info("Batch size: %s", BATCH_SIZE)
    logger.info("Max workers: %s", MAX_WORKERS)
    df = load_dataset()
    texts = df["text"].tolist()
    original_labels = df["label"].tolist()

    clients = build_clients()
    enabled_llms = get_enabled_llms(clients)
    classifier_model, classifier_vectorizer = load_classifier_artifacts()

    for llm_name in enabled_llms:
        for prompt_key, prompt in PROMPTS.items():
            logger.info(
                "Starting generation | llm=%s | prompt=%s | total_texts=%s",
                llm_name,
                prompt_key,
                len(texts),
            )

            new_texts = []
            start_time = time.time()
            total_fallbacks = 0

            for batch_start in range(0, len(texts), BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, len(texts))
                batch_rows = [
                    {"row_number": index, "text": text}
                    for index, text in enumerate(texts[batch_start:batch_end], start=batch_start + 1)
                ]

                try:
                    batch_results, batch_fallbacks = process_batch(
                        batch_rows,
                        prompt,
                        llm_name,
                        clients,
                        prompt_key,
                    )
                except FatalProviderError as error:
                    logger.error(
                        "Stopping provider due to fatal configuration error | llm=%s | prompt=%s | batch_start_row=%s | error=%s",
                        llm_name,
                        prompt_key,
                        batch_start + 1,
                        error,
                    )
                    logger.error(
                        "Skipping remaining prompts for %s until the credential issue is fixed",
                        llm_name,
                    )
                    break

                new_texts.extend(batch_results)
                total_fallbacks += batch_fallbacks

                completed = len(new_texts)
                logger.info(
                    "Batch completed | llm=%s | prompt=%s | rows=%s-%s | completed=%s/%s | batch_fallbacks=%s | total_fallbacks=%s",
                    llm_name,
                    prompt_key,
                    batch_start + 1,
                    batch_end,
                    completed,
                    len(texts),
                    batch_fallbacks,
                    total_fallbacks,
                )

                time.sleep(REQUEST_DELAY_SECONDS)
            else:
                predicted_generated_labels = predict_labels(
                    new_texts,
                    classifier_model,
                    classifier_vectorizer,
                )
                label_changed = [
                    int(predicted_label != original_label)
                    for predicted_label, original_label in zip(predicted_generated_labels, original_labels)
                ]

                new_df = pd.DataFrame(
                    {
                        "original_text": texts,
                        "generated_text": new_texts,
                        "original_label": original_labels,
                        "predicted_generated_label": predicted_generated_labels,
                        "label_changed": label_changed,
                        "prompt_key": prompt_key,
                        "llm_name": llm_name,
                    }
                )

                output_path = os.path.join(OUTPUT_DIR, f"{llm_name}_{prompt_key}.csv")
                new_df.to_csv(output_path, index=False)

                elapsed = time.time() - start_time
                logger.info(
                    "Saved generated dataset | llm=%s | prompt=%s | path=%s | elapsed=%.2fs | total_fallbacks=%s",
                    llm_name,
                    prompt_key,
                    output_path,
                    elapsed,
                    total_fallbacks,
                )
                continue

            break

    original_output_path = os.path.join(OUTPUT_DIR, "original.csv")
    df.to_csv(original_output_path, index=False)
    logger.info("Saved original sampled dataset to %s", original_output_path)
    logger.info("All datasets generated successfully")


if __name__ == "__main__":
    main()
