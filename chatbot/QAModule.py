from utils.utilities import Utils

"""Retrieve answer from QA CSV by checking
 similarity of a query with QA questions"""
class QAModule:

    def __init__(self, df):
        # Extract questions and answers from csv
        self.questions = df["Question"].astype(str).tolist()
        self.answers = df["Answer"].astype(str).tolist()
        self.utils = Utils()

        # Vectorise question documents
        self.question_vectors = self.utils.vectorise(self.questions)

    """Find the answer best matching the question (using search engine logic)"""
    def get_answer(self, query: str):
        best_idx = self.utils.find_best_match(query, self.question_vectors, "cosine", 0.7)

        if best_idx is None:
            return "I'm not sure I have the answer to that. Try rephrasing."

        return self.answers[best_idx]