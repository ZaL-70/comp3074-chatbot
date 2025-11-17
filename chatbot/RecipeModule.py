import random
from data.recipe_data import RECIPES, KNOWN_DIETS

"""Manages queries related to transactions (recipes)"""
# Notes:
# - Add stemming to ings/diet (flexible input to data match)
# - Fix recipe start to start queried recipe
# - Add ability to ask user between multiple/single existing preference before recipe start
# - Add half-way marker to step-by-step guide (or any extras similar)
class RecipeModule:
    def __init__(self):
        self.recipes = RECIPES
        self.diet_keywords = KNOWN_DIETS.union(self.extract_all_diets())
        self.user_saved = []
        self.active_recipe = None
        self.preferred_recipe = None
        self.step_index = 0

    # --- Transactional Functions ---

    # Select a random recipe
    def random_recipe(self):
        self.preferred_recipe = random.choice(list(self.recipes.keys()))
        return self.preferred_recipe

    # Specific Recipe Search (ingredients and dietary)
    def find_recipes(self, diet_filters=None, ingredient_filters=None, or_query=False):
        # Exit early if filters are both empty
        if not (diet_filters or ingredient_filters):
            return "No recipes match those ingredients or diet requirements."

        # First check "or" word used
        if or_query:
            or_matches = [
                name for name, recipe in self.recipes.items()
                if self.match_or(recipe, ingredient_filters, diet_filters, fallback=False)
            ]
            # Standard response ("or" word was specified)
            return (f"I found these recipes- {', '.join(or_matches)}" or
                    "No recipes match those ingredients or diet requirements.")

        # Default behaviour: AND logic first (unless "or" word was seen)
        and_matches = [
            name for name, recipe in self.recipes.items()
            if self.match_and(recipe, ingredient_filters, diet_filters)
        ]
        if and_matches:
            return f"I found these recipes- {', '.join(and_matches)}"

        # (Potential refactor: Before fallback to basic or, check ingredients
        # & diet filter used -> ingredients did not exist with those diets)
        # Otherwise fallback to OR logic (if no matches)
        or_matches = [
            name for name, recipe in self.recipes.items()
            if self.match_or(recipe, ingredient_filters, diet_filters, fallback=True)
        ]
        if or_matches:  # Transparent message in fall back case
            return f"I don't think I have that. Here's some potential alternatives- {', '.join(or_matches)}"

        return "No recipes match those ingredients or diet requirements."

    # Recipe Steps Intent
    def start_recipe_steps(self):
        if self.preferred_recipe not in self.recipes:
            return "I couldn’t find that recipe."
        self.active_recipe = self.preferred_recipe
        self.step_index = 0
        return f"Let's make {self.active_recipe}! Step 1: {self.recipes[self.active_recipe]['steps'][0]} (type 'next' to continue)"

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
            return f"You're done! Enjoy your {recipe_done}!"

    # Recipe Save Intent
    def save_recipe(self, recipe_name):
        if recipe_name not in self.recipes:
            return "I can only save recipes I know."
        self.user_saved.append(recipe_name)
        return f"Recipe '{recipe_name}' saved!"

    # View Saved Recipes Intent
    def recall_saved(self):
        if not self.user_saved:
            return "You haven’t saved any recipes yet."
        return "You’ve saved: " + ", ".join(self.user_saved)

    # --- Helper Functions ----

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

        print(ingredient_filters)
        print(diet_filters)
        return diet_filters, ingredient_filters
