import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))


def clean_text(text):
    """
    Basic text preprocessing
    """

    text = text.lower()

    text = re.sub(r'[^\w\s]', '', text)

    words = text.split()

    words = [w for w in words if w not in stop_words]

    return " ".join(words)


def preprocess_dataframe(df):

    df["clean_text"] = df["text"].apply(clean_text)

    return df