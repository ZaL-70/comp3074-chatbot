import re
import nltk
from nltk import word_tokenize, pos_tag

# Install necessary nltk resources
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

"""Detect proper nouns (names) preceding 
introductory words to learn the users name"""
class IdentityManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        # Keywords that often precede a name
        self.user_name = None
        self.intro_keywords = [
            "name is", "names", "call me", "i'm", "im", "i am", "this is", "its", "it's"
        ]
        self._initialized = True

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
            self.user_name = name
            return name

        # Else fallback to potential names (in lowercase) preceding intro words
        after = text_lower.split(phrase, 1)[1].strip()
        if not after or len(after.split()) == 0:
            return "invalid"

        candidate = after.split()[0]
        candidate = "".join(c for c in candidate if c.isalpha())

        # If candidate contains digits, invalid
        if any(char.isdigit() for char in after.split()[0]):
            return "has_number"

        if candidate:
            self.user_name = candidate.title()
            return self.user_name

        return "invalid"

    NON_NAME_RESPONSES = {
        "dont", "do", "not", "know", "i dont know", "idk",
        "no", "maybe", "later", "nothing", "none", "no idea",
        "whatever", "anything", "something", "meh", "fine",
        "you", "choose", "your", "choice", "up", "to",
        "not sure", "unsure"
    }

    # Helper to check if "quick, single response" input follows naming conventions
    def looks_like_name(self, text: str):
        # Long phrases are not names
        words = text.strip().lower().split()
        if len(words) == 0 or len(words) > 4:
            return "invalid"

        # Reject any phrase or word in the NON_NAME_RESPONSES list
        if text in self.NON_NAME_RESPONSES:
            return "anonymous"
        if any(w in self.NON_NAME_RESPONSES for w in words):
            return "anonymous"

        # Reject punctuation (except apostrophes)
        if re.search(r"[^a-zA-Z\s']", text):
            return "invalid"

        # If all words alphabetic-like, consider it a name
        for w in words:
            if not re.match(r"^[A-Za-z']+$", w):
                return "has_number"

        return "valid"