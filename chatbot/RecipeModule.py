import random
from data.recipe_data import RECIPES, KNOWN_DIETS
from enum import Enum, auto

# Recipe query specific state
class RecipeState(Enum):
    DEFAULT = auto()
    SEARCHING = auto()
    COOKING = auto()
    RECIPE_CONFIRMING = auto()
    STEPS_FINISH = auto()

"""Manages queries related to transactions (recipes)"""
# Notes:
# - Add stemming/lemmatising to ings/diet (flexible input to data match)
# - Add half-way marker to step-by-step guide or any similar features
#   (conversational markers)
# - Add context tracking on steps (how many left, start, end etc)
class RecipeModule:
    def __init__(self):
        self.recipes = RECIPES
        self.diet_keywords = KNOWN_DIETS.union(self.extract_all_diets())
        self.user_saved = []
        self.active_recipe = None
        self.preferred_recipe = None
        self.search_matches = []
        self.step_index = 0
        self.state = RecipeState.DEFAULT

    # --- Transactional Functions ---

    # Select a random recipe
    def random_recipe(self):
        self.preferred_recipe = random.choice(list(self.recipes.keys()))
        self.state = RecipeState.SEARCHING
        self.search_matches = [self.preferred_recipe]
        return self.preferred_recipe + "\nDo you want me to guide you through the steps for this?"

    # Specific Recipe Search (ingredients and dietary)
    def find_recipes(self, diet_filters=None, ingredient_filters=None, or_query=False):
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
            message = (f"I found these recipes- {', '.join(matches)}." or
                    message)

        # Default behaviour: AND logic first (unless "or" word was seen)
        if not or_query:
            matches = [
                name for name, recipe in self.recipes.items()
                if self.match_and(recipe, ingredient_filters, diet_filters)
            ]
            if matches:
                message = f"I found these recipes- {', '.join(matches)}."
        # Fall back to or query to try and get some result
        if not matches:
            matches = [
                name for name, recipe in self.recipes.items()
                if self.match_or(recipe, ingredient_filters, diet_filters, fallback=True)
            ]
            if matches:  # Transparent message in fall back case
                message = f"I don't think I have that. Here's some potential alternatives- {', '.join(matches)}."

        if matches:
            self.state = RecipeState.SEARCHING
            self.search_matches = matches

        if len(matches) == 1:
            self.preferred_recipe = matches[0]
            return message + "\nDo you want me to guide you through the steps for this?"
        if len(matches) > 1:
            return message + "\nDo you want me to guide you through the steps for one of these?"

        return message

    # Recipe Steps Intent
    def start_recipe_steps(self, recipe_name=None):
        recipe, err = self.resolve_recipe(recipe_name)
        if err:
            return err
        self.active_recipe = recipe
        self.step_index = 0
        first_step = self.recipes[self.active_recipe]['steps'][0]
        self.state = RecipeState.COOKING
        return f"Let's make {self.active_recipe}! Step 1: {first_step} (type 'next' to continue)"

    def next_step(self):
        if not self.active_recipe:
            return "No active recipe. Start one first!"
        self.step_index += 1
        steps = self.recipes[self.active_recipe]['steps']
        if self.step_index < len(steps):
            return f"Step {self.step_index+1}: {steps[self.step_index]} (type 'next' to continue)"
        else:
            recipe_done = self.active_recipe
            self.active_recipe = None
            self.state = RecipeState.STEPS_FINISH
            return (f"You're done! Enjoy your {recipe_done}!"
                    f"\nIf you want I can save this recipe, would you like me to save it?")

    # Recipe Save Intent
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

    # View Saved Recipes Intent
    def recall_saved(self):
        if not self.user_saved:
            return "Nothing haven’t saved any recipes yet."
        return ", ".join(self.user_saved)

    # --- Helper Functions ----

    REFERENCE_WORDS = {"that", "that one", "it", "this", "those", "them"}

    # Returns the user_name of a recipe from a query
    def extract_recipe_name(self, user_input, intent=None):
        user_input = user_input.lower().strip()
        for recipe_name in self.recipes.keys():
            if recipe_name.lower() in user_input:
                return recipe_name
        # Potential addition - look for fewer words part of full recipe user_name
        # Check if reference word used to refer to a recipe
        intents_requiring_recipe = {
            "recipe_steps",
            "recipe_save",
        }
        if intent in intents_requiring_recipe:
            normalized = " ".join(user_input.split())
            for ref in self.REFERENCE_WORDS:
                if normalized.endswith(ref):
                    if self.preferred_recipe:
                        return self.preferred_recipe
                    else:
                        return "NO_TARGET_REF"

        return "UNKNOWN_RECIPE"

    def resolve_recipe(self, recipe_name=None):
        self.preferred_recipe = recipe_name
        if recipe_name == "UNKNOWN_RECIPE":
            return None, "I don't recognise that recipe. Could you tell me the exact name?"

        if (not self.preferred_recipe) or (recipe_name == "NO_TARGET_REF"):
            return None, "I’m not sure which recipe you mean. Could you tell me the name?"

        if self.preferred_recipe not in self.recipes:
            return None, f"I couldn’t find the recipe '{self.preferred_recipe}'"

        return self.preferred_recipe, None

    # Checks if every filter word is in a recipe (preferred, default behaviour)
    @staticmethod
    def match_and(recipe, ingredient_filters, diet_filters):
        ingredient_ok = all(
            any(f in ingredient for ingredient in recipe["ingredients"])
            for f in ingredient_filters
        ) if ingredient_filters else True

        diet_ok = all(
            f in recipe["diet"]
            for f in diet_filters
        ) if diet_filters else True

        return ingredient_ok and diet_ok

    # Checks any filter words are in a recipe (& handle special fallback cases)
    @staticmethod
    def match_or(recipe, ingredient_filters, diet_filters, fallback=False):
        if fallback and diet_filters and not ingredient_filters: # Cases where no diets exist (so nothing can possibly match)
            return False
        elif fallback and diet_filters: # Cases where diets exist but didn't with the ingredients (e.g. veg options but not with chicken)
            diet_ok = any(
                f in recipe["diet"]
                for f in diet_filters
            ) if diet_filters else False
            return diet_ok
        elif fallback and not diet_filters: # Cases where ingredients exist but not with each other (e.g. milk & spices)
            ingredient_ok = any(
                any(f in ingredient for ingredient in recipe["ingredients"])
                for f in ingredient_filters
            ) if ingredient_filters else False
            return ingredient_ok
        else: # Valid cases that don't fall back
            ingredient_ok = any(
                any(f in ingredient for ingredient in recipe["ingredients"])
                for f in ingredient_filters
            ) if ingredient_filters else False
            diet_ok = any(
                f in recipe["diet"]
                for f in diet_filters
            ) if diet_filters else False
            return ingredient_ok or diet_ok

    # Extracts diets from recipe data
    def extract_all_diets(self):
        diet_set = set()
        for recipe in self.recipes.values():
            for diet in recipe.get("diet", []):
                diet_set.add(diet.lower())
        return diet_set

    # Extract diet/ingredients mentioned in (filter) query into separate lists
    def extract_filters(self, filters):
        diet_filters = []
        ingredient_filters = []

        for word in filters:
            # match diet words
            if word in self.diet_keywords:
                diet_filters.append(word)
                continue

            # match ingredients
            for recipe in self.recipes.values():
                if any(word == ingredient for ingredient in recipe["ingredients"]):
                    ingredient_filters.append(word)
                    break

        return diet_filters, ingredient_filters
