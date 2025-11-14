from chatbot.core import Chatbot

def main():
    bot = Chatbot()
    print("Hello, this is COMP3074 HAI CW Bot. How can I help?.\n")

    # Main chatbot loop
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Bot: Goodbye!")
            break
        response = bot.respond(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()