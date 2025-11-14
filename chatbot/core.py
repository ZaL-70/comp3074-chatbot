import random
import pandas as pd
from chatbot.QAModule import QAModule
from chatbot.SmalltalkMatcher import SmalltalkMatcher
from chatbot.identities import *
from chatbot.IntentMatcher import *
from data.responses import *

"""Brings separate components of the chatbot together 
 so it can respond to queries accordingly"""
class Chatbot:

    def __init__(self):
        self.df = pd.read_csv("data/COMP3074-CW1-Dataset.csv")
        self.intent_model = IntentMatcher()
        self.qa_module = QAModule(self.df)
        self.identity_manager = IdentityManager()
        self.smalltalk_module = SmalltalkMatcher(self.identity_manager)

    def respond(self, user_input: str):
        # Check name assignment
        name = self.identity_manager.extract_name(user_input)

        # Respond to name assignment prompt
        if name:
            return f"Nice to meet you, {name}!"

        # Predict intent
        intent = self.intent_model.predict_intent(user_input)

        # Respond to certain intents with a relevant response
        if intent == "ask_name":
            if self.identity_manager.name:
                return f"Your name is {self.identity_manager.name}."
            return "I don't know your name yet. What should I call you?"
        elif intent == "capabilities":
            return random.choice(INTENT_RESPONSES["capabilities"])
        elif intent == "qa":
            return self.qa_module.get_answer(user_input)
        elif intent == "small_talk":
            return self.smalltalk_module.get_response(user_input)
        else:
            return random.choice(INTENT_RESPONSES["unknown"])