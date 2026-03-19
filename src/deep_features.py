from sentence_transformers import SentenceTransformer
import os
import joblib


FEATURE_DIR = r"C:\Users\dell\PycharmProjects\LLM-Evaluation-MasterThesis\Result\Features"
os.makedirs(FEATURE_DIR, exist_ok=True)


def extract_sbert_and_save(texts, model_name):

    print(f"\nExtracting SBERT features: {model_name}")

    model = SentenceTransformer(model_name)

    embeddings = model.encode(texts, show_progress_bar=True)

    file_path = os.path.join(
        FEATURE_DIR,
        f"{model_name.replace('/', '_')}.pkl"
    )

    joblib.dump(embeddings, file_path)

    print("Saved:", file_path)

    return embeddings