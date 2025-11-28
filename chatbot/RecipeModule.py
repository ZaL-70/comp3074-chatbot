import random
import nltk
from nltk.stem import PorterStemmer
from chatbot.NLGPipeline import NLGPipeline
from data.recipe_data import RECIPES, KNOWN_DIETS
from enum import Enum, auto
from utils.TextMatcher import TextMatcher

nltk.download("punkt", quiet=True)

# Recipe query specific states
class RecipeState(Enum):
    DEFAULT = auto()
    SEARCHING = auto()
    RECIPE_CONFIRMING = auto()
    STEPS_FINISH = auto()

"""Manages queries related to transactions (recipes)"""
class RecipeModule:
    def __init__(self):
        self.recipes = RECIPES
        self.NLG = NLGPipeline()
        self.stemmer = PorterStemmer()
        self.preprocess_recipes()
        self.recipe_training = {name: [name] for name in self.recipes.keys()}
        self.recipe_matcher = TextMatcher(self.recipe_training,False,1,2)
        self.diet_keywords = KNOWN_DIETS.union(self.extract_all_diets())
        self.user_saved = []
        self.active_recipe = None
        self.preferred_recipe = None
        self.search_matches = []
        self.step_index = 0
        self.state = RecipeState.DEFAULT

    # --- Transactional Functions ---

    # Return a random recipe with description
    def random_recipe(self):
        self.preferred_recipe = random.choice(list(self.recipes.keys()))
        self.state = RecipeState.SEARCHING
        self.search_matches = [self.preferred_recipe]
        description = self.NLG.generate_recipe_description(self.preferred_recipe, "overview")
        return (f"Here's a recipe for you: {description}\n"
                "Do you want me to guide you through the steps for this?")

    # Specific Recipe Search (ingredients & dietary)
    def find_recipes(self, diet_filters=None, ingredient_filters=None, or_query=False):
        # Assume no matches have been found first
        message = "No recipes match those ingredients or diet requirements."
        matches = []
        # Exit early if filters are both empty
        if not (diet_filters or ingredient_filters):
            return message

        # First check "or" word used
        if or_query:
            matches = [
                name for name, recipe in self.recipes.items()
                if self.match_or(recipe, ingredient_filters, diet_filters, fallback=False)
            ]
            # Standard response ("or" word was specified)
            message = self.NLG.aggregate_recipe_list(matches, "exact")

        # Default behaviour: AND logic first (unless "or" word was already seen)
        if not or_query:
            matches = [
                name for name, recipe in self.recipes.items()
                if self.match_and(recipe, ingredient_filters, diet_filters)
            ]
            if matches:
                message = self.NLG.aggregate_recipe_list(matches, "exact")

        # Fall back to or query to try & get some form of results
        if not matches:
            matches = [
                name for name, recipe in self.recipes.items()
                if self.match_or(recipe, ingredient_filters, diet_filters, fallback=True)
            ]
            if matches:  # Transparent message in fall back case
                message = self.NLG.aggregate_recipe_list(matches, "alternative")

        if matches:
            self.state = RecipeState.SEARCHING
            self.search_matches = matches

        # Use proper grammar depending on results
        if len(matches) == 1:
            self.preferred_recipe = matches[0]
            return (message +
                    "\nDo you want me to guide you through the steps for this?")
        if len(matches) > 1:
            self.state = RecipeState.RECIPE_CONFIRMING
            return (message +
                    "\nDo you want me to guide you through the steps for one of these?")

        return message

    # Begin steps for a given recipe
    def start_recipe_steps(self, recipe_name=None):
        recipe, err = self.resolve_recipe(recipe_name)
        if err:
            return err
        self.active_recipe = recipe
        self.step_index = 0
        first_step = self.recipes[self.active_recipe]['steps'][0]
        return (f"Let's make {self.active_recipe}! "
                f"Step 1: {first_step} (type 'next' to continue)")

    # Moves recipe steps forward
    def next_step(self):
        if not self.active_recipe:
            return "No active recipe. Start one first!"

        self.step_index += 1
        steps = self.recipes[self.active_recipe]['steps']

        # Conversational & contextual markers
        if self.step_index < len(steps):
            remaining = len(steps) - self.step_index
            if self.step_index == len(steps) - 1:
                marker = "Finally, "
            elif self.step_index == len(steps) // 2:
                marker = "We're halfway there! Next, "
            else:
                marker = "Next, "

            step_text = steps[self.step_index].lower()
            progress = f"({remaining} step{'s' if remaining > 1 else ''} remaining)"

            return f"{marker}{step_text} {progress}"
        else:
            recipe_done = self.active_recipe
            self.state = RecipeState.STEPS_FINISH
            self.active_recipe = None
            self.preferred_recipe = None # Reset after finishing steps
            # Contextual marker
            return (f"You're done! Enjoy your {recipe_done}!"
                    f"\nIf you want I can save this recipe, would you like me to save it?")

    # Saves a recipe in a user defined list for future reference
    def save_recipe(self, recipe_name):
        recipe, err = self.resolve_recipe(recipe_name)
        if err:
            return err
        self.state = RecipeState.DEFAULT
        # Avoid duplicates
        if recipe not in self.user_saved:
            self.user_saved.append(recipe)
            return f"Recipe '{recipe}' saved!"
        return "Looks like you've got this one saved already"

    # Return list of user's saved recipes
    def recall_saved(self):
        # Select template with most suitable grammar based on the current list
        if not self.user_saved:
            return "You haven't saved any recipes yet."
        if len(self.user_saved) == 1:
            return f"You have one saved recipe: {self.user_saved[0]}"
        elif len(self.user_saved) == 2:
            return f"Your saved recipes: {self.user_saved[0]} and {self.user_saved[1]}"
        else:
            recipe_list = ', '.join(self.user_saved[:-1]) + ', and ' + self.user_saved[-1]
            return f"Your saved recipes: {recipe_list}"

    # --- Helper Functions ----

    # Applies stemming to recipe data creating new, separate columns
    def preprocess_recipes(self):
        for name, recipe in self.recipes.items():
            # Stem ingredients
            recipe["stemmed_ingredients"] = [
                self.stemmer.stem(ing.lower()) for ing in recipe["ingredients"]
            ]
            # Stem diets
            recipe["stemmed_diet"] = [
                self.stemmer.stem(d.lower()) for d in recipe["diet"]
            ]
            # Stem recipe name words
            recipe["stemmed_name"] = [
                self.stemmer.stem(w) for w in name.lower().split()
            ]

    REFERENCE_WORDS = {"that", "that one", "it", "this", "those", "them"}
    # Returns the name of a recipe from a query
    def extract_recipe_name(self, user_input, intent=None):
        user_input = user_input.lower().strip()

        # Check exact match
        for recipe_name in self.recipes.keys():
            if recipe_name.lower() in user_input:
                return recipe_name

        # Check stem based match
        stems = [self.stemmer.stem(token) for token in nltk.word_tokenize(user_input)]
        for recipe_name, recipe in self.recipes.items():
            if any(stem in recipe["stemmed_name"] for stem in stems):
                return recipe_name

        # Cosine similarity based match
        best = self.recipe_matcher.predict_category(user_input, threshold=0.55)
        if best:
            return best

        # Check if reference word used to refer to a recipe
        intents_requiring_recipe = {
            "recipe_steps",
            "recipe_save",
            "recipe_details"
        }
        if intent in intents_requiring_recipe:
            normalized = " ".join(user_input.split())
            for ref in self.REFERENCE_WORDS:
                if normalized.endswith(ref):
                    if self.preferred_recipe:
                        return self.preferred_recipe
                    else:
                        return "NO_TARGET_REF"

        # Else recipe is completely unknown
        return "UNKNOWN_RECIPE"

    # Return according recipe name assignment (reset name or set to correct one)
    # & return relevant error message when querying recipe (names)
    def resolve_recipe(self, recipe_name=None):
        # Explicit recipe search for non-existent recipe (reset)
        if recipe_name == "UNKNOWN_RECIPE":
            self.state = RecipeState.DEFAULT
            return None, "I couldn't find that recipe"

        # Referred recipe search on a non-existent referral target (reset, use confirmation state)
        if recipe_name == "NO_TARGET_REF":
            self.state = RecipeState.RECIPE_CONFIRMING
            return None, "I’m not sure which recipe you mean. Could you tell me the name?"

        # Searched for a correct, existing recipe (set as current preference, no error message)
        if recipe_name in self.recipes:
            self.preferred_recipe = recipe_name
            return self.preferred_recipe, None

        # Default to non-existent recipe
        return None, f"I couldn’t find the recipe '{recipe_name}'"

    # Checks if EVERY filter word is in a recipe
    @staticmethod
    def match_and(recipe, ingredient_filters, diet_filters):
        ingredient_ok = all(
            any(f in ingredient for ingredient in recipe["stemmed_ingredients"])
            for f in ingredient_filters
        ) if ingredient_filters else True

        diet_ok = all(
            f in recipe["stemmed_diet"]
            for f in diet_filters
        ) if diet_filters else True

        return ingredient_ok and diet_ok

    # Checks if ANY filter words are in a recipe
    # Handles special fallback cases for most discoverability (tries to return some result always)
    @staticmethod
    def match_or(recipe, ingredient_filters, diet_filters, fallback=False):
        # Cases where no diets exist (so nothing can possibly match)
        if fallback and diet_filters and not ingredient_filters:
            return False
        # Cases where diets exist but didn't with the ingredients (e.g. vegetarian options but not with chicken)
        elif fallback and diet_filters:
            diet_ok = any(
                f in recipe["stemmed_diet"]
                for f in diet_filters
            ) if diet_filters else False
            return diet_ok
        # Cases where ingredients exist but not with each other (e.g. recipes with both milk & spices)
        elif fallback and not diet_filters:
            ingredient_ok = any(
                any(f in ingredient for ingredient in recipe["stemmed_ingredients"])
                for f in ingredient_filters
            ) if ingredient_filters else False
            return ingredient_ok
        # Valid cases that don't fall back
        else:
            ingredient_ok = any(
                any(f in ingredient for ingredient in recipe["stemmed_ingredients"])
                for f in ingredient_filters
            ) if ingredient_filters else False
            diet_ok = any(
                f in recipe["stemmed_diet"]
                for f in diet_filters
            ) if diet_filters else False
            return ingredient_ok or diet_ok

    # Extracts diets from existing recipe data returning them in a set
    def extract_all_diets(self):
        diet_set = set()
        for recipe in self.recipes.values():
            for diet in recipe.get("diet", []):
                diet_set.add(diet.lower())
        return diet_set

    # Extract diet/ingredients mentioned in a query returning them as separate lists
    # Ingredients based on recipes, diet based on an arbitrary list & recipes
    def extract_filters(self, filters):
        diet_filters = []
        ingredient_filters = []

        stemmed_filters = [self.stemmer.stem(word.lower()) for word in filters]

        for stem in stemmed_filters:
            # Match (stemmed) diet words to inputted filters
            if stem in {self.stemmer.stem(diet_word) for diet_word in self.diet_keywords}:
                diet_filters.append(stem)
                continue

            # Match (stemmed) ingredient words to inputted filters
            for recipe in self.recipes.values():
                if stem in recipe["stemmed_ingredients"]:
                    ingredient_filters.append(stem)
                    break

        return diet_filters, ingredient_filters
