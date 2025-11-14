import random
from chatbot.identities import IdentityManager
from data.responses import SMALL_TALK_RESPONSES, INTENT_RESPONSES
from data.smalltalk_training import SMALL_TALK_TRAINING
from utils.TextMatcher import TextMatcher

"""Match small talk prompts to its relevant category by training
on pre-anticipated small talk phrases (labelled by category)"""
class SmalltalkMatcher(TextMatcher):

    def __init__(self, identity_manager: IdentityManager):
        super().__init__(SMALL_TALK_TRAINING, use_stopwords=True)

        # Data required for small talk responses
        self.identity_manager = identity_manager
        self.responses = SMALL_TALK_RESPONSES

    """Give relevant response for the type of small talk"""
    def get_response(self, query: str):
        category = self.find_category(query, metric="cosine", threshold=0.8)

        if category == "greetings":
            if self.identity_manager.name:
                return f"Hello again, {self.identity_manager.name}!"
            return random.choice(SMALL_TALK_RESPONSES["greetings"])
        elif category == "weather":
            return random.choice(SMALL_TALK_RESPONSES["weather"])
        elif category == "mood":
            return random.choice(SMALL_TALK_RESPONSES["mood"])
        elif category == "thanks":
            return random.choice(SMALL_TALK_RESPONSES["thanks"])
        else:
            return random.choice(INTENT_RESPONSES["unknown"])