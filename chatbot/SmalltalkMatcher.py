import random
from data.responses import SMALL_TALK_RESPONSES, INTENT_RESPONSES
from utils.TextMatcher import TextMatcher

"""Match small talk prompts to its relevant category by training
on pre-anticipated small talk phrases (labelled by category)"""
class SmalltalkMatcher(TextMatcher):

    def __init__(self, intent_df):
        small_talk_training = (intent_df[intent_df["intent"] == "small_talk"]
            .groupby("sub_intent")["text"]
            .apply(list).to_dict())
        super().__init__(small_talk_training, use_stopwords=False, analyzer="char_wb", ngram_min=1, ngram_max=2)

        # Data required for small talk responses
        self.responses = SMALL_TALK_RESPONSES

    # Give relevant response for the type of small talk (using similarity (search engine) logic) (Lab 1)
    def get_response(self, query: str):
        category = self.predict_category(query, threshold=0.7)

        if category == "weather":
            return random.choice(SMALL_TALK_RESPONSES["weather"])
        elif category == "mood":
            return random.choice(SMALL_TALK_RESPONSES["mood"])
        elif category == "thanks":
            return random.choice(SMALL_TALK_RESPONSES["thanks"])
        else:
            return random.choice(INTENT_RESPONSES["UNKNOWN"])