import random
from data.recipe_data import RECIPES

"""Library of NLG methods for every step of the pipeline to
 generate relevant descriptions about recipes & search queries"""
class NLGPipeline:
    def __init__(self):
        self.recipes = RECIPES

    # Decide which fields to include (& level of detail). Can change based on user intent
    def content_determination(self, recipe_name, purpose="overview"):
        recipe = self.recipes[recipe_name]

        if purpose == "overview": # (Default)
            return {
                'name': recipe_name,
                'ingredients': recipe['ingredients'][:3],  # Top 3 ingredients only
                'diet': recipe['diet'],
                'step_count': len(recipe['steps'])
            }

    # Creates a structure for the data before aggregation using existing content
    @staticmethod
    def document_structuring(content, purpose="overview"):
        if purpose == "overview":
            return {
                'introduction': [content['name'], content['step_count']],
                'highlights': content['ingredients'],
                'dietary': content['diet']
            }

        return content

    # Generates main template the user will see in the output
    @staticmethod
    def aggregation(structured_content, purpose="overview"):
        aggregated = {}

        if purpose == "overview":
            # Aggregate ingredients into a natural list
            ingredients = structured_content['highlights']
            if len(ingredients) > 1:
                ing_text = ', '.join(ingredients[:-1]) + ' and ' + ingredients[-1]
            else:
                ing_text = ingredients[0]

            # Aggregate quick intro & main ingredients
            aggregated['intro'] = (f"{structured_content['introduction'][0]}"
                                   f" requires {structured_content['introduction'][1]} steps")
            aggregated['highlights'] = f"featuring {ing_text}"

            # Aggregate dietary info
            diet_labels = structured_content['dietary']
            if diet_labels:
                if len(diet_labels) > 1:
                    diet_text = ', '.join(diet_labels[:-1]) + ' and ' + diet_labels[-1]
                else:
                    diet_text = diet_labels[0]
                aggregated['dietary'] = f"suitable for {diet_text} diets"
            else:
                aggregated['dietary'] = ""

        return aggregated

    # Return relevant template based on search query results
    # - Default case - exact matches otherwise assume no precise matches were found
    @staticmethod
    def aggregate_recipe_list(matches, match_type):
        recipe_list = ', '.join(matches[:-1]) + ', and ' + matches[-1]
        if len(matches) == 1:
            if match_type == "exact":
                return f"I found the perfect match: {matches[0]}"
            else:
                return f"Here's a potential alternative: {matches[0]}"
        elif len(matches) > 2:
            if match_type == "exact":
                return f"I found these recipes: {recipe_list}"
            else:
                return f"I don't think I have anything with all of those. Here are some alternatives: {recipe_list}"
        else:
            if match_type == "exact":
                return f"I found these recipes: {recipe_list}"
            else:
                return f"I don't think I have anything with all of those. Here are some alternatives: {recipe_list}"

    # Replaces words from the template. Can decide based on experience level
    # (e.g. beginner, experienced, expert etc.) & with randomness
    @staticmethod
    def lexical_choice(aggregated_content, user_experience="beginner"):
        # Adjust vocabulary based on user experience
        if user_experience == "beginner":
            # Use simpler terms
            aggregated_content['intro'] = aggregated_content['intro'].replace('requires', 'needs')

        # Add variety to avoid repetition
        if 'featuring' in aggregated_content.get('highlights', ''):
            alternatives = ['featuring', 'with', 'including', 'containing']
            chosen = random.choice(alternatives)
            aggregated_content['highlights'] = aggregated_content['highlights'].replace('featuring', chosen)

        return aggregated_content

    # Replaces recipe names with referral words. Words decided by some existing context
    # (e.g. just mentioned, no context, mentioned a very long time ago etc.)
    @staticmethod
    def referring_expression(content, context=None):
        # If recipe was just mentioned, use pronouns
        if context and context.get('just_mentioned'):
            content['intro'] = content['intro'].replace(context['recipe_name'], 'It')

        return content

    # Put templates together with correct language grammar & flow
    @staticmethod
    def realisation(lexical_content):
        # Combine all parts with proper punctuation & flow
        intro = lexical_content['intro'].capitalize()
        parts = [intro]

        if 'highlights' in lexical_content and lexical_content['highlights']:
            parts.append(lexical_content['highlights'])
        if 'dietary' in lexical_content and lexical_content['dietary']:
            parts.append(lexical_content['dietary'])

        # Join with commas & proper sentence structure
        if len(parts) == 1:
            return parts[0] + "."
        elif len(parts) == 2:
            return f"{parts[0]}, {parts[1]}."
        else:
            # Use discourse markers for better flow
            return f"{parts[0]}, {parts[1]}, and is {parts[2]}."

    # Full pipeline
    def generate_recipe_description(self, recipe_name, purpose="overview", user_experience="beginner", context=None):
        content = self.content_determination(recipe_name, purpose)
        structured = self.document_structuring(content, purpose)
        aggregated = self.aggregation(structured, purpose)
        lexical = self.lexical_choice(aggregated, user_experience)
        referred = self.referring_expression(lexical, context)
        return self.realisation(referred)