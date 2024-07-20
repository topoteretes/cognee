import re
import nltk
from nltk.tag import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def extract_topics_yake(texts: list[str]):
    from yake import KeywordExtractor

    keyword_extractor = KeywordExtractor(
        top = 3,
        n = 2,
        dedupLim = 0.2,
        dedupFunc = "levenshtein", # "seqm" | "levenshtein"
        windowsSize = 1,
    )

    for text in texts:
        topics = keyword_extractor.extract_keywords(preprocess_text(text))
        yield [topic[0] for topic in topics]

def extract_topics_keybert(texts: list[str]):
    from keybert import KeyBERT

    kw_model = KeyBERT()

    for text in texts:
        topics = kw_model.extract_keywords(
            preprocess_text(text),
            keyphrase_ngram_range = (1, 2),
            top_n = 3,
            # use_mmr = True,
            # diversity = 0.9,
        )
        yield [topic[0] for topic in topics]

def preprocess_text(text: str):
    try:
        # Used for stopwords removal.
        stopwords.ensure_loaded()
    except LookupError:
        nltk.download("stopwords", quiet = True)
        stopwords.ensure_loaded()

    try:
        # Used in WordNetLemmatizer.
        wordnet.ensure_loaded()
    except LookupError:
        nltk.download("wordnet", quiet = True)
        wordnet.ensure_loaded()

    try:
        # Used in word_tokenize.
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet = True)

    text = text.lower()

    # Remove punctuation
    text = re.sub(r"[^\w\s-]", "", text)

    # Tokenize the text
    tokens = word_tokenize(text)

    tagged_tokens = pos_tag(tokens)
    tokens = [word for word, tag in tagged_tokens if tag in ["NNP", "NN", "JJ"]]

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize the text
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join tokens back to a single string
    processed_text = " ".join(tokens)

    return processed_text


# def clean_text(text: str):
#     text = re.sub(r"[ \t]{2,}|[\n\r]", " ", text.lower())
#     # text = re.sub(r"[`\"'.,;!?…]", "", text).strip()
#     return text

# def remove_stop_words(text: str):
#     try:
#         stopwords.ensure_loaded()
#     except LookupError:
#         download("stopwords")
#         stopwords.ensure_loaded()

#     stop_words = set(stopwords.words("english"))
#     text = text.split()
#     text = [word for word in text if not word in stop_words]
#     return " ".join(text)


if __name__ == "__main__":
    import os

    file_dir = os.path.dirname(os.path.realpath(__file__))

    with open(os.path.join(file_dir, "texts.json"), "r", encoding = "utf-8") as file:
        import json
        texts = json.load(file)

    for topics in extract_topics_yake(texts):
        print(topics)
        print("\n")
