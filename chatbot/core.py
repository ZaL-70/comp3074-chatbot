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
        self.recipes_handler = RecipeModule()
        self.user_state = UserState.DEFAULT
        self.pending_intent = None
        self.RECIPE_INTENTS = {
            "recipe_steps",
            "recipe_save",
            "recipe_recall",
            "recipe_search",
            "recipe_filter_search"
        }
        self.AGREE_TERMS = {
            "yes", "yep", "yeah", "yup", "sure", "absolutely", "of course",
            "definitely", "ok", "okay", "sure thing", "affirmative", "please do",
            "go ahead", "sounds good", "why not", "alright", "do it"
        }
        self.DISAGREE_TERMS = {
            "no", "nope", "nah", "not really", "don't", "do not",
            "negative", "no thanks", "no thank you", "stop", "cancel",
            "please don't", "rather not", "not now", "leave it"
        }

    def respond(self, user_input: str):
        # ---- Respond to user_name assignment prompt if applicable ----
        name = self.identity_manager.extract_name(user_input)
        if name == "has_number":
            self.identity_manager.user_name = None
            return "Names shouldn’t contain numbers. What should I call you?"
        if name == "invalid":
            self.identity_manager.user_name = None
            return "That doesn’t look like a name. What should I call you?"
        if name:
            return f"Nice to meet you, {self.identity_manager.user_name}!"

        # Predict intent
        intent = self.intent_model.predict(user_input)
        print("intent: ", intent)

        # Reset states when topic switches away from recipe (intent not matching with current recipe state)
        if self.recipes_handler.state != RecipeState.DEFAULT and intent not in self.RECIPE_INTENTS:
            self.recipes_handler.state = RecipeState.DEFAULT
            self.recipes_handler.active_recipe = None
            self.recipes_handler.search_matches = []
            self.recipes_handler.preferred_recipe = None
            self.pending_intent = None

        # ---- Respond to recipe state specific situations when applicable ----
        if self.recipes_handler.active_recipe and (
                user_input.lower() == "next" or "continue" in user_input.lower()):
            return self.recipes_handler.next_step()
        if self.recipes_handler.state == RecipeState.SEARCHING:
            if user_input.lower() == "yes" and len(self.recipes_handler.search_matches) == 1:
                return self.recipes_handler.start_recipe_steps(self.recipes_handler.search_matches[0])
            elif user_input.lower() == "no":
                self.recipes_handler.state = RecipeState.DEFAULT
                return "No worries. Is there anything else I can assist with"
            else:
                self.recipes_handler.state = RecipeState.DEFAULT
        # Only run in non-recipe-search intent (preserve chronological transaction order)
        if self.recipes_handler.state == RecipeState.RECIPE_CONFIRMING and intent != "recipe_filter_search":
            if user_input.lower().strip() in self.AGREE_TERMS and len(self.recipes_handler.search_matches) > 1:
                return "Great! Which one do you want the steps for?"
            elif user_input.lower().strip() in self.DISAGREE_TERMS:
                self.recipes_handler.state = RecipeState.DEFAULT
                return "No worries. Is there anything else I can assist with"
            else:
                recipe_name = self.recipes_handler.extract_recipe_name(
                    user_input,
                    self.pending_intent
                )
                if self.pending_intent == "recipe_save":
                    self.pending_intent = None
                    return self.recipes_handler.save_recipe(recipe_name)
                self.pending_intent = None
                return self.recipes_handler.start_recipe_steps(recipe_name)
        if self.recipes_handler.state == RecipeState.STEPS_FINISH:
            if user_input.lower().strip() in self.AGREE_TERMS and len(self.recipes_handler.search_matches) == 1:
                return self.recipes_handler.save_recipe(self.recipes_handler.search_matches[0]) # Add behaviour
            if user_input.lower().strip() in self.DISAGREE_TERMS:
                self.recipes_handler.state = RecipeState.DEFAULT
                return "No worries. Is there anything else I can assist with"

        # ---- Allow single phrase response for user asking personal identity ----
        if self.user_state == UserState.NAME_ASKING and intent == "UNKNOWN":
            name_check = self.identity_manager.looks_like_name(user_input)
            # Error checking with guided recovery
            if name_check == "valid":
                self.identity_manager.user_name = user_input.title()
                self.user_state = UserState.DEFAULT
                return f"Nice to meet you, {self.identity_manager.user_name}!"
            elif name_check == "has_number":    # Has numbers, ask again
                return "Names shouldn’t contain numbers. What should I call you?"
            elif name_check == "anonymous":
                self.identity_manager.user_name = "My Friend"
                self.user_state = UserState.DEFAULT
                return "Okay, I'll just call you 'My Friend' for now."
            else:   # Invalid, ask again
                return "That doesn’t look like a name. What should I call you?"

        # ---- Respond to certain intents with a relevant response ----
        if intent == "recipe_steps":
            self.pending_intent = intent
            recipe_name = self.recipes_handler.extract_recipe_name(user_input, intent)
            return self.recipes_handler.start_recipe_steps(recipe_name)
        if intent == "recipe_save":
            self.pending_intent = intent
            recipe_name = self.recipes_handler.extract_recipe_name(user_input, intent)
            return self.recipes_handler.save_recipe(recipe_name)
        if intent == "recipe_recall":
            if self.identity_manager.user_name:
                return  (f"{self.identity_manager.user_name}'s favourites: " +
                     self.recipes_handler.recall_saved())
            return "Your favourites: " + self.recipes_handler.recall_saved()
        if intent == "recipe_search":
            return f"Here's a recipe for you: {self.recipes_handler.random_recipe()}"
        if intent == "recipe_filter_search":
            filters = [word.lower() for word in user_input.split()]
            diet_filters, ingredient_filters = self.recipes_handler.extract_filters(filters)
            or_query = "or" in user_input.lower()
            matches = self.recipes_handler.find_recipes(diet_filters, ingredient_filters, or_query)
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