from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline

"""Intent match prompt by training on
 pre-anticipated queries (labelled by intent)"""
# Notes:
# - Fix classification (confuses, search with recipe start & "continue" with recipe start)
#   - E.g. add obvious phrase checks (improved classification accuracy & evaluate separately
#   - Or augment dataset for improved classification (see Lecture 16)
class IntentClassifier:
    def __init__(self, df):
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(1,3))),
            ('clf', SGDClassifier(loss='log_loss', alpha=0.0001))
        ])

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        self.scores = cross_val_score(self.model, df["text"], df["intent"], cv=cv)

        self.model.fit(df['text'], df['intent'])

    def predict(self, user_input):
        probs = self.model.predict_proba([user_input])[0]
        max_prob = max(probs)
        # Debug
        print("Intent Probability:", max_prob)
        print("Accuracy scores:", self.scores)
        print("Mean accuracy:", self.scores.mean())
        if max_prob < 0.6:
            return "UNKNOWN"
        return self.model.predict([user_input])[0]