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
            "name", "call", "i'm", "im", "i am", "this is", "its", "it's"
        ]

    # Lab 0: Text processing with pos tags inspired
    def extract_name(self, text: str):
        text_lower = text.lower()

        # Check sentence talks about a name or identity
        if not any(k in text_lower for k in self.intro_keywords):
            return None

        # Tokenize & POS-tag sentence
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)

        # Try to find proper nouns (NNP tags)
        candidates = [word for word, tag in tagged if tag == "NNP"]

        if not candidates:
            return None

        # Combine consecutive proper nouns
        name = " ".join(candidates)

        if name:
            self.name = name.strip().title()
            return name.strip().title()
        else:
            return None