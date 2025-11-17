import nltk
from nltk import word_tokenize, pos_tag

# Install necessary nltk resources
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

"""Detect proper nouns (names) preceding 
introductory words to learn the users name"""
class IdentityManager:

    def __init__(self):
        # Keywords that often precede a name
        self.name = None
        self.intro_keywords = [
            "name is", "names", "call me", "i'm", "im", "i am", "this is", "its", "it's"
        ]

    def contains_intro_phrase(self, text_lower):
        tokens = text_lower.split()

        for phrase in self.intro_keywords:
            phrase_tokens = phrase.split()
            # sliding window over tokens
            for i in range(len(tokens) - len(phrase_tokens) + 1):
                if tokens[i:i + len(phrase_tokens)] == phrase_tokens:
                    return phrase

        return None

    # Lab 0 - text processing with pos tags inspired
    def extract_name(self, text: str):
        text_lower = text.lower()

        # Check sentence uses self intro words
        phrase = self.contains_intro_phrase(text_lower)
        if not phrase:
            return None

        # Tokenize & pos tag sentence
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)

        # Try to find proper nouns (NNP tags)
        proper_nouns = [word for word, tag in tagged if tag == "NNP"]

        if proper_nouns:
            name = " ".join(proper_nouns).title()
            self.name = name
            return name

        # Else fallback to potential names (in lowercase) preceding intro words
        after = text_lower.split(phrase, 1)[1].strip()
        if not after or len(after.split()) == 0:
            return None

        candidate = after.split()[0]
        candidate = "".join(c for c in candidate if c.isalpha())

        if candidate:
            name = candidate.title()
            self.name = name
            return name

        return None