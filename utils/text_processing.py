from sklearn.feature_extraction.text import TfidfVectorizer


def build_note_features(notes):

    notes = notes.fillna("").astype(str)

    vectorizer = TfidfVectorizer(
        max_features=500,
        lowercase=True,
        stop_words="english"
    )

    X_notes = vectorizer.fit_transform(notes)

    return X_notes, vectorizer


#baseline 3
from sklearn.feature_extraction.text import TfidfVectorizer


def build_note_features(notes):

    notes = notes.fillna("").astype(str)

    vectorizer = TfidfVectorizer(
        max_features=500,
        lowercase=True,
        stop_words="english"
    )

    X_notes = vectorizer.fit_transform(notes)

    return X_notes, vectorizer