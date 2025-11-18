import random
import pandas as pd
from chatbot.QAModule import QAModule
from chatbot.RecipeModule import RecipeModule
from chatbot.SmalltalkMatcher import SmalltalkMatcher
from chatbot.identities import *
from chatbot.IntentMatcher import *
from data.responses import *

"""Brings separate components of the chatbot together 
 so it can respond to queries accordingly"""
class Chatbot:

    def __init__(self):
        self.qa_df = pd.read_csv("data/COMP3074-CW1-Dataset.csv")
        self.intent_df = pd.read_csv("data/intent-training.csv")
        self.intent_model = IntentClassifier(self.intent_df)
        self.qa_module = QAModule(self.qa_df)
        self.identity_manager = IdentityManager()
        self.smalltalk_module = SmalltalkMatcher(self.intent_df, self.identity_manager)
        self.recipes = RecipeModule()

    def respond(self, user_input: str):
        # Check name assignment
        name = self.identity_manager.extract_name(user_input)

        # Respond to name assignment prompt
        if name:
            return f"Nice to meet you, {name}!"

        # Predict intent
        intent = self.intent_model.predict(user_input)
        print(intent)

        # Respond to certain intents with a relevant response
        if intent == "recipe_search":
            return f"Here's a recipe for you: {self.recipes.random_recipe()}"
        elif intent == "recipe_filter_search":
            filters = [word.lower() for word in user_input.split()]
            diet_filters, ingredient_filters = self.recipes.extract_filters(filters)
            or_query = "or" in user_input.lower()
            matches = self.recipes.find_recipes(diet_filters, ingredient_filters, or_query)
            return matches
        elif intent == "recipe_steps":
            recipe_name = self.recipes.extract_recipe_name(user_input)
            print("Recipe input:", recipe_name)
            return self.recipes.start_recipe_steps(recipe_name)
        elif user_input.lower() == "next":
            return self.recipes.next_step()
        elif intent == "recipe_save":
            return self.recipes.save_recipe("pancakes")
        elif intent == "recipe_recall":
            return self.recipes.recall_saved()
        elif intent == "greeting":
            if self.identity_manager.name:
                return f"Hello again, {self.identity_manager.name}!"
            return random.choice(INTENT_RESPONSES["greeting"])
        elif intent == "ask_name":
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