from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

"""Intent match prompt by training on
 pre-anticipated queries (labelled by intent)"""
# NoteL Add obvious phrase checks (improved classification accuracy)
# Evaluate separately
class IntentClassifier:
    def __init__(self, df):

        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(3,5))),
            ('clf', LinearSVC())
        ])

        self.scores = cross_val_score(self.model, df['text'], df['intent'], cv=5)

        self.model.fit(df['text'], df['intent'])

    def predict(self, user_input):
        print("Accuracy scores:", self.scores)
        print("Mean accuracy:", self.scores.mean())
        return self.model.predict([user_input])[0]