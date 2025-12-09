import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from chatbot.RecipeModule import RecipeModule

"""Intent match prompt using rules or by training on
 pre-anticipated queries (labelled by intent)"""
class IntentClassifier:
    def __init__(self, df):
        self.recipe_handler = RecipeModule()
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(2,5))),
            ('clf', SGDClassifier(loss='log_loss', class_weight="balanced", alpha=0.0005))
        ])

        self.model.fit(df['text'], df['intent'])

    # Defines intents identifiable by rules & no ML techniques
    def rule_based_intents(self, user_input):
        # Recipe save rule - Matches: "save pancakes", "save the soup", "save my recipe" etc.
        if re.match(r"^save\b", user_input.lower().strip()):
            return "recipe_save"

        # Clarification rule - Matches ambiguous recipe input like "pancakes"
        if len(user_input.split()) < 4:
            recipe_name = self.recipe_handler.extract_recipe_name(user_input)
            if recipe_name not in ["UNKNOWN_RECIPE", "NO_TARGET_REF"]:
                return "clarification_intent"

        return None  # No rule matched

    # Predict user's intent using predefined rules & a classification model
    def predict(self, user_input):
        # Try rule based first
        rule_intent = self.rule_based_intents(user_input)
        if rule_intent is not None:
            return rule_intent

        # Otherwise default to classification model
        probs = self.model.predict_proba([user_input])[0]
        max_prob = max(probs)
        # Use a confidence level to ignore potential vague inputs
        if max_prob < 0.35:
            return "UNKNOWN"
        return self.model.predict([user_input])[0]