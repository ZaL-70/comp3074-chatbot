import nltk
import numpy as np
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

# Download necessary nltk resources
nltk.download('stopwords', quiet=True)

"""Utility class for low-level, general, 
query-doc matching using TF-IDF"""
class TfIdfMatcher:
    def __init__(self, ngram_min: int = 1, ngram_max: int = 1, use_stopwords: bool = True):
        # Prepare TD-IDF
        if use_stopwords:
            self.count_vect = TfidfVectorizer(stop_words=stopwords.words('english'), ngram_range=(ngram_min, ngram_max))
        else:
            self.count_vect = TfidfVectorizer(analyzer='char_wb', ngram_range=(ngram_min, ngram_max))

        self.tfidf_transformer = TfidfTransformer(use_idf=True, sublinear_tf=True)

    # Vectorise documents using TF-IDF
    def vectorise(self, documents):
        x_counts = self.count_vect.fit_transform(documents)
        document_vectors = self.tfidf_transformer.fit_transform(x_counts)

        return document_vectors

    # Low-Level document matching for a query (e.g. QA answer index)
    def find_best_match(self, query: str, vectors, threshold: float = 0.7):
        # Vectorize the query
        q_vec = self.count_vect.transform([query])
        q_tfidf = self.tfidf_transformer.transform(q_vec)

        # Compute similarities
        sims = cosine_similarity(q_tfidf, vectors).flatten()

        # Get the top match
        best_idx = np.argmax(sims)
        best_score = sims[best_idx]
        # Debug
        # print("similarity score: ", best_score)

        if best_score < threshold:
            return None

        return best_idx