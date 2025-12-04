import re

from nltk import PorterStemmer

from utils.TfIdfMatcher import TfIdfMatcher

"""Retrieve answer from QA CSV by checking
 similarity of a query with QA questions"""
class QAModule:

    def __init__(self, qa_df):
        # Extract questions & answers from QA csv
        self.questions = [self.preprocess(q) for q in qa_df["Question"].astype(str).tolist()]
        self.answers = qa_df["Answer"].astype(str).tolist()
        self.tfidf_matcher = TfIdfMatcher(use_stopwords=True)

        # Vectorise question documents
        self.question_vectors = self.tfidf_matcher.vectorise(self.questions)

    # Find the answer best matching the question (using similarity (search engine) logic) (Lab 1)
    def get_answer(self, query: str):
        query_processed = self.preprocess(query)
        best_idx, best_score = self.tfidf_matcher.find_best_match(query_processed, self.question_vectors, 0.7)

        if best_idx is None or best_score <= 0.2:
            return "no_answer"
        if 0.2 < best_score < 0.6:
            return  "You're asking a question right, can you be more specific?"

        return self.answers[best_idx]

    # Stemming helper
    @staticmethod
    def preprocess(query: str) -> str:
        stemmer = PorterStemmer()
        text = query.lower().strip()
        # Remove punctuation but keep apostrophes (help contractions)
        text = re.sub(r"[^a-z0-9\s']", " ", text)
        tokens = text.split()
        stemmed = [stemmer.stem(tok) for tok in tokens]
        return " ".join(stemmed)
