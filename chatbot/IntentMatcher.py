from utils.TextMatcher import TextMatcher
from data.intent_training import INTENTS_TRAINING

"""Intent match prompt by training on
 pre-anticipated queries (labelled by intent)"""
class IntentMatcher(TextMatcher):

    def __init__(self):
        super().__init__(INTENTS_TRAINING, use_stopwords=False)

    def predict_intent(self, user_input: str):
        intent = self.find_category(user_input, metric="euclidean", threshold=0.5)

        if intent is None:
            return "unknown"

        return intent