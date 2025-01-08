import re
from nltk.downloader import download
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def extract_topics(text: str):
    sentences = sent_tokenize(text)

    try:
        wordnet.ensure_loaded()
    except LookupError:
        download("wordnet")
        wordnet.ensure_loaded()

    lemmatizer = WordNetLemmatizer()
    base_notation_sentences = [lemmatizer.lemmatize(sentence) for sentence in sentences]

    tf_vectorizer = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None)
    transformed_corpus = tf_vectorizer.fit_transform(base_notation_sentences)

    svd = TruncatedSVD(n_components=10)
    svd_corpus = svd.fit(transformed_corpus)

    feature_scores = dict(zip(tf_vectorizer.vocabulary_, svd_corpus.components_[0]))

    topics = sorted(
        feature_scores,
        # key = feature_scores.get,
        key=lambda x: transformed_corpus[0, tf_vectorizer.vocabulary_[x]],
        reverse=True,
    )[:10]

    return topics


def clean_text(text: str):
    text = re.sub(r"[ \t]{2,}|[\n\r]", " ", text.lower())
    return re.sub(r"[`\"'.,;!?…]", "", text).strip()


def remove_stop_words(text: str):
    try:
        stopwords.ensure_loaded()
    except LookupError:
        download("stopwords")
        stopwords.ensure_loaded()

    stop_words = set(stopwords.words("english"))
    text = text.split()
    text = [word for word in text if word not in stop_words]
    return " ".join(text)


if __name__ == "__main__":
    text = """Lorem Ipsum is simply dummy text of the printing and typesetting industry... Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book… It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.
        Why do we use it?
        It is a long established fact that a reader will be distracted by the readable content of a page when looking at its layout! The point of using Lorem Ipsum is that it has a more-or-less normal distribution of letters, as opposed to using 'Content here, content here', making it look like readable English. Many desktop publishing packages and web page editors now use Lorem Ipsum as their default model text, and a search for 'lorem ipsum' will uncover many web sites still in their infancy. Various versions have evolved over the years, sometimes by accident, sometimes on purpose (injected humour and the like).
    """
    print(extract_topics(remove_stop_words(clean_text(text))))
