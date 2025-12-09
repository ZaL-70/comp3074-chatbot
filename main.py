import nltk
from chatbot.core import Chatbot

def main():
    # Install necessary nltk resources
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

    bot = Chatbot()
    print("Hey, this is ChefBot your personal kitchen companion! How may I assist?\n")

    # Main chatbot loop
    while True:
        user_input = input("You: ")
        if user_input.lower().strip() in ["quit", "exit", "bye"]:
            print("Bot: Goodbye!")
            break
        response = bot.respond(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()