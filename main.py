import nltk

from chatbot.core import Chatbot

def main():
    # Install necessary nltk resources
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

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