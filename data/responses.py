from chatbot.identities import IdentityManager

identity_manager = IdentityManager()

def get_named_greeting():
    name = identity_manager.user_name
    return [
        f"Hey there {name}",
        f"Hello again {name}",
        f"Hi {name}! Good to see you!"
    ]

INTENT_RESPONSES = {
    "greeting": [
        "Hello there! What's your name?",
        "Hi! Nice to meet you. What should I call you?",
    ],
    "capabilities": [
        "I can be your personal kitchen assistant helping you decide"
        " what to cook, guide you through the process, \nstore your "
        "favourite recipes & answer cooking related queries. "
        "I can also remember your name & have a short chat!"
    ],
    "ask_bot_name": [
        "I'm ChefBot, your personal kitchen companion!",
        "Call me ChefBot!",
        "It's ChefBot. Lets cook!"
    ],
    "UNKNOWN": [
        "I'm not sure I understand. Could you rephrase that?"
    ]
}

SMALL_TALK_RESPONSES = {
    "weather": [
        "I hope it's sunny where you are!",
        "It's always perfect weather in cyberspace.",
        "Rainy days are perfect for chatting with bots like me!"
    ],
    "mood": [
        "I'm just code, but I'm running great!",
        "I'm doing just fine, thanks for asking!",
        "Doing well, thanks for asking!",
        "I'm here and ready to help!"
    ],
    "thanks": [
        "You're very welcome!",
        "No problem!",
        "Happy to help!"
    ]
}