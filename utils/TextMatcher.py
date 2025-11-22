from utils.TfIdfMatcher import TfIdfMatcher

"""Base class for high-level text matcher classes using TF-IDF"""
class TextMatcher:

    def __init__(self, training_data: dict, use_stopwords: bool = False):
        self.texts = None
        self.labels = None
        self.vectors = None
        self.tfidf_matcher = TfIdfMatcher(use_stopwords=use_stopwords)

        # Anticipated queries
        self.training_data = training_data

        # Vectorise & label training data
        self.train()

    """Vectorize and label training data"""
    def train(self):
        texts = []
        labels = []
        for category, phrases in self.training_data.items():
            for phrase in phrases:
                texts.append(phrase)
                labels.append(category)

        # Vectorize documents
        self.vectors = self.tfidf_matcher.vectorise(texts)
        self.texts = texts
        self.labels = labels

    """Find the best matching category for a query (e.g. intent, 
     small talk)"""
    def predict_category(self, query: str, threshold: float = 0.7):
        query_clean = query.strip().lower()

        # Check for exact text match first
        # (failsafe for phrases with high amount of stop words)
        for idx, text in enumerate(self.texts):
            if query_clean == text:
                return self.labels[idx]

        # Otherwise use cosine similarity
        best_idx = self.tfidf_matcher.find_best_match(query, self.vectors, threshold)

        if best_idx is None:
            return None

        return self.labels[best_idx]