import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_keywords(text: str) -> list[str]:
    tokens = nltk.word_tokenize(text)

    tags = nltk.pos_tag(tokens)
    nouns = [word for (word, tag) in tags if tag == "NN"]

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(nouns)

    top_nouns = sorted(
        vectorizer.vocabulary_,
        key = lambda x: tfidf[0, vectorizer.vocabulary_[x]],
        reverse = True
    )

    keywords = []

    for word in top_nouns:
        if len(word) > 3:
            keywords.append(word)
        if len(keywords) >= 15:
            break

    return keywords
