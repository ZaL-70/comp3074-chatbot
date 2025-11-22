import random
from enum import Enum, auto
import pandas as pd
from data.responses import *
from chatbot.QAModule import QAModule
from chatbot.RecipeModule import RecipeModule, RecipeState
from chatbot.SmalltalkMatcher import SmalltalkMatcher
from chatbot.identities import IdentityManager
from chatbot.IntentMatcher import *

# General user specific states
class UserState(Enum):
    DEFAULT = auto()
    NAME_ASKING = auto()

"""Brings separate components of the chatbot together 
 so it can respond to queries accordingly"""
class Chatbot:
    def __init__(self):
        self.qa_df = pd.read_csv("data/COMP3074-CW1-Dataset.csv")
        self.intent_df = pd.read_csv("data/intent-training.csv")
        self.intent_model = IntentClassifier(self.intent_df)
        self.qa_module = QAModule(self.qa_df)
        self.identity_manager = IdentityManager()
        self.smalltalk_module = SmalltalkMatcher(self.intent_df)
        self.recipes = RecipeModule()
        self.user_state = UserState.DEFAULT

    def respond(self, user_input: str):
        # Respond to user_name assignment prompt
        if self.identity_manager.extract_name(user_input):
            return f"Nice to meet you, {self.identity_manager.user_name}!"

        # Predict intent
        intent = self.intent_model.predict(user_input)
        print(intent)

        # Respond to state specific situations when applicable
        if self.recipes.state == RecipeState.COOKING and self.recipes.active_recipe and (
                user_input.lower() == "next" or "continue" in user_input.lower()):
            return self.recipes.next_step()
        if self.recipes.state == RecipeState.SEARCHING:
            if user_input.lower() == "yes" and len(self.recipes.search_matches) == 1:
                return self.recipes.start_recipe_steps(self.recipes.search_matches[0])
            if user_input.lower() == "yes" and len(self.recipes.search_matches) > 1:
                self.recipes.state = RecipeState.RECIPE_CONFIRMING
                return "Great! Which one do you want the steps for?" # Add new function
            if user_input.lower() == "no":
                return "No worries. Is there anything else I can assist with"
        if self.recipes.state == RecipeState.STEPS_FINISH:
            if user_input.lower() == "yes" and len(self.recipes.search_matches) == 1:
                return self.recipes.save_recipe(self.recipes.search_matches[0]) # Add behaviour
            if user_input.lower() == "no":
                self.recipes.state = RecipeState.DEFAULT
                return "No worries. Is there anything else I can assist with"
        if self.recipes.state == RecipeState.RECIPE_CONFIRMING:
            if user_input.lower() in self.recipes.search_matches:
                return self.recipes.start_recipe_steps(user_input)
        if self.user_state == UserState.NAME_ASKING and intent == "UNKNOWN":
            self.identity_manager.user_name = user_input.title()
            self.user_state = UserState.DEFAULT # Reset state after use
            return f"Nice to meet you, {self.identity_manager.user_name}!"

        # Respond to certain intents with a relevant response
        if intent == "recipe_steps":
            recipe_name = self.recipes.extract_recipe_name(user_input, intent)
            return self.recipes.start_recipe_steps(recipe_name)
        if intent == "recipe_save":
            recipe_name = self.recipes.extract_recipe_name(user_input, intent)
            return f"{self.identity_manager.user_name}'s favourites: " + self.recipes.save_recipe(recipe_name)
        if intent == "recipe_recall":
            return self.recipes.recall_saved()
        if intent == "recipe_search":
            return f"Here's a recipe for you: {self.recipes.random_recipe()}"
        if intent == "recipe_filter_search":
            filters = [word.lower() for word in user_input.split()]
            diet_filters, ingredient_filters = self.recipes.extract_filters(filters)
            or_query = "or" in user_input.lower()
            matches = self.recipes.find_recipes(diet_filters, ingredient_filters, or_query)
            return matches
        if intent == "greeting":
            if self.identity_manager.user_name:
                return random.choice(get_named_greeting())
            self.user_state = UserState.NAME_ASKING
            return random.choice(INTENT_RESPONSES["greeting"])
        if intent == "ask_user_name":
            if self.identity_manager.user_name:
                return f"Your name is {self.identity_manager.user_name}."
            self.user_state = UserState.NAME_ASKING
            return "I don't know your name yet. What should I call you?"
        if intent == "ask_bot_name":
            return random.choice(INTENT_RESPONSES["ask_bot_name"])
        if intent == "capabilities":
            return random.choice(INTENT_RESPONSES["capabilities"])
        if intent == "qa":
            return self.qa_module.get_answer(user_input)
        if intent == "small_talk":
            return self.smalltalk_module.get_response(user_input)
        if intent == "UNKNOWN":
            return random.choice(INTENT_RESPONSES["UNKNOWN"])