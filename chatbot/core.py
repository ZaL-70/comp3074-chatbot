import random
import pandas as pd
from enum import Enum, auto
from chatbot.NLGPipeline import NLGPipeline
from data.responses import *
from chatbot.QAModule import QAModule
from chatbot.RecipeModule import RecipeState
from chatbot.SmalltalkMatcher import SmalltalkMatcher
from chatbot.identities import IdentityManager
from chatbot.IntentMatcher import *

# General user specific states
class UserState(Enum):
    DEFAULT = auto()
    RECIPE_INTENT_CLARIFYING = auto()
    SELF_NAME_ASSIGNING = auto()

"""Brings separate components of the chatbot together 
 so it can respond to queries accordingly"""
class Chatbot:
    def __init__(self):
        self.qa_df = pd.read_csv("data/cooking-qa.csv")
        self.intent_df = pd.read_csv("data/intent-training.csv")
        self.intent_model = IntentClassifier(self.intent_df)
        self.qa_module = QAModule(self.qa_df)
        self.identity_manager = IdentityManager()
        self.smalltalk_module = SmalltalkMatcher(self.intent_df)
        self.recipes_handler = RecipeModule()
        self.NLG = NLGPipeline()
        self.nlg_context = {
            "just_mentioned": False,
            "recipe_name": None
        }
        self.user_state = UserState.DEFAULT
        self.pending_intent = None
        self.pending_recipe = None
        self.RECIPE_INTENTS = {
            "recipe_steps",
            "recipe_save",
            "recipe_recall",
            "recipe_search",
            "recipe_filter_search",
            "recipe_details",
            "clarification_intent"
        }
        self.AGREE_TERMS = {
            "yes", "yep", "yeah", "yup", "sure", "absolutely", "of course",
            "definitely", "ok", "okay", "sure thing", "sure", "please do",
            "go ahead", "sounds good", "why not", "alright", "do it"
        }
        self.DISAGREE_TERMS = {
            "no", "nope", "nah", "not really", "dont", "do not",
            "negative", "no thanks", "no thank you", "stop", "cancel",
            "please don't", "rather not", "not now", "leave it"
        }

    # Choose or create template with response to return to user whilst keeping
    # track of & updating context, user states & possible pending intents
    def respond(self, user_input: str):
        # ---- Predict intent ----
        intent = self.intent_model.predict(user_input)
        # Debug
        print("intent: ", intent)

        # ---- Respond to user_name assignment prompt if applicable ----
        name = self.identity_manager.extract_name(user_input)
        if name == "has_number":
            self.identity_manager.user_name = None
            self.user_state = UserState.SELF_NAME_ASSIGNING
            return "Names shouldn't contain numbers. What should I call you?"
        if name == "invalid":
            self.identity_manager.user_name = None
            self.user_state = UserState.SELF_NAME_ASSIGNING
            return "That doesn't look like a name. What should I call you?"
        if name:
            self.user_state = UserState.DEFAULT
            return f"Nice to meet you, {self.identity_manager.user_name}!"

        # ---- Reset states when topic switches ----
        # For recipe states
        if self.recipes_handler.state != RecipeState.DEFAULT and intent not in self.RECIPE_INTENTS and intent != "UNKNOWN":
            self.recipes_handler.state = RecipeState.DEFAULT
            self.recipes_handler.active_recipe = None
            self.recipes_handler.search_matches = []
            self.recipes_handler.preferred_recipe = None
            self.pending_intent = None
            self.nlg_context = {"just_mentioned": False, "recipe_name": None}

        # Debug
        print("recipe state: ", self.recipes_handler.state)

        # For general user states
        if self.user_state == UserState.SELF_NAME_ASSIGNING and intent not in {"UNKNOWN", "ask_user_name"}:
            self.user_state = UserState.DEFAULT

        # ---- Respond to state specific situations when applicable ----
        # Recipe state situations
        if self.recipes_handler.active_recipe and (
                user_input.lower() == "next" or "continue" in user_input.lower()):
            return self.recipes_handler.next_step()
        if self.recipes_handler.state == RecipeState.SEARCHING:
            if user_input.lower().strip() in self.AGREE_TERMS and len(self.recipes_handler.search_matches) == 1:
                return self.recipes_handler.start_recipe_steps(self.recipes_handler.search_matches[0])
            elif user_input.lower().strip() in self.DISAGREE_TERMS:
                self.recipes_handler.state = RecipeState.DEFAULT
                return "No worries. Is there anything else I can assist with"
            else:
                self.recipes_handler.state = RecipeState.DEFAULT
        if self.recipes_handler.state == RecipeState.RECIPE_CONFIRMING and intent != "recipe_filter_search":
            if user_input.lower().strip() in self.AGREE_TERMS and len(self.recipes_handler.search_matches) > 1:
                self.pending_intent = "recipe_steps"
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
                if self.pending_intent == "recipe_steps":
                    self.pending_intent = None
                    return self.recipes_handler.start_recipe_steps(recipe_name)
                if self.pending_intent == "recipe_details":
                    recipe, err = self.recipes_handler.resolve_recipe(recipe_name)
                    if err:
                        return err
                    self.pending_intent = None
                    return self.NLG.generate_recipe_description(recipe_name, "overview", context=self.nlg_context)
        if self.recipes_handler.state == RecipeState.STEPS_FINISH:
            self.recipes_handler.state = RecipeState.DEFAULT
            if user_input.lower().strip() in self.AGREE_TERMS and self.recipes_handler.preferred_recipe is not None:
                return self.recipes_handler.save_recipe(self.recipes_handler.preferred_recipe)
            if user_input.lower().strip() in self.DISAGREE_TERMS:
                return "No worries. Is there anything else I can assist with"

        # User state specific situations
        if self.user_state == UserState.RECIPE_INTENT_CLARIFYING:
            text = user_input.lower().strip()
            if any(word in text for word in ["step", "steps", "instructions"]):
                self.user_state = UserState.DEFAULT
                return self.recipes_handler.start_recipe_steps(self.pending_recipe)
            if any(word in text for word in ["save", "keep", "store", "favourite", "favorite"]):
                self.user_state = UserState.DEFAULT
                return self.recipes_handler.save_recipe(self.pending_recipe)
            if any(word in text for word in ["describe", "description", "details", "overview", "info", "information"]):
                self.user_state = UserState.DEFAULT
                return self.NLG.generate_recipe_description(
                    self.pending_recipe,
                    "overview",
                    context=self.nlg_context
                )
        if self.user_state == UserState.SELF_NAME_ASSIGNING and intent == "UNKNOWN":
            name_check = self.identity_manager.looks_like_name(user_input)
            # Error checking with guided recovery
            if name_check == "valid":
                self.identity_manager.user_name = user_input.title()
                self.user_state = UserState.DEFAULT
                return f"Nice to meet you, {self.identity_manager.user_name}!"
            elif name_check == "has_number":    # Has numbers, ask again
                return "Names shouldn't contain numbers. What should I call you?"
            elif name_check == "anonymous":
                self.identity_manager.user_name = "My Friend"
                self.user_state = UserState.DEFAULT
                return "Okay, I'll just call you 'My Friend' for now."
            else:   # Invalid, ask again
                return "That doesn't look like a name. What should I call you?"

        # ---- Respond to certain intents with a relevant response ----
        if intent == "clarification_intent":
            recipe_name = self.recipes_handler.extract_recipe_name(user_input)
            recipe, err = self.recipes_handler.resolve_recipe(recipe_name)
            if err:
                return err
            # Store for referring expressions
            self.pending_recipe = recipe_name
            self.nlg_context = {
                "just_mentioned": True,
                "recipe_name": recipe_name
            }
            self.user_state = UserState.RECIPE_INTENT_CLARIFYING
            return f"You're asking about {recipe_name}. Did you want the steps, a description, or to save it?"
        if intent == "recipe_details":
            self.pending_intent = intent
            recipe_name = self.recipes_handler.extract_recipe_name(user_input, intent)
            recipe, err = self.recipes_handler.resolve_recipe(recipe_name)
            if err:
                return err
            self.nlg_context = { "just_mentioned": True, "recipe_name": recipe_name }
            description = self.NLG.generate_recipe_description(recipe_name, "overview", context=self.nlg_context)
            return description
        if intent == "recipe_steps":
            self.pending_intent = intent
            recipe_name = self.recipes_handler.extract_recipe_name(user_input, intent)
            self.nlg_context = { "just_mentioned": True, "recipe_name": recipe_name }
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
            return self.recipes_handler.random_recipe()
        if intent == "recipe_filter_search":
            clean_input = user_input
            for char in [',', ';']:
                clean_input = clean_input.replace(char, ' ')
            filters = [word.lower() for word in clean_input.split()]
            diet_filters, ingredient_filters = self.recipes_handler.extract_filters(filters)
            or_query = "or" in user_input.lower()
            matches = self.recipes_handler.find_recipes(diet_filters, ingredient_filters, or_query)
            return matches
        if intent == "greeting":
            if self.identity_manager.user_name:
                return random.choice(get_named_greeting())
            self.user_state = UserState.SELF_NAME_ASSIGNING
            return random.choice(INTENT_RESPONSES["greeting"])
        if intent == "ask_user_name":
            if self.identity_manager.user_name:
                return f"Your name is {self.identity_manager.user_name}."
            self.user_state = UserState.SELF_NAME_ASSIGNING
            return "I don't know your name yet. What should I call you?"
        if intent == "ask_bot_name":
            return random.choice(INTENT_RESPONSES["ask_bot_name"])
        if intent == "capabilities":
            return random.choice(INTENT_RESPONSES["capabilities"])
        if intent == "small_talk":
            return self.smalltalk_module.get_response(user_input)

        # ---- Respond to QA if no intent matched ----
        answer = self.qa_module.get_answer(user_input)
        if answer != "no_answer" and intent == "UNKNOWN":
            return answer

        if intent == "UNKNOWN":
            return random.choice(INTENT_RESPONSES["UNKNOWN"])