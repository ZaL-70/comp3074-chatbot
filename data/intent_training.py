import pandas as pd
from data.smalltalk_training import SMALL_TALK_TRAINING

# Flatten categorised small talk data into 1D unlabelled
small_talk_flat = []
for phrases in SMALL_TALK_TRAINING.values():
    for phrase in phrases:
        small_talk_flat.append(phrase)

df = pd.read_csv("data/COMP3074-CW1-Dataset.csv")

INTENTS_TRAINING = {
            "ask_name": [
                "what is my name", "whats my name",
                "do you know my name", "tell me my name"],
            "capabilities": ["what can you do", "help me"],
            "qa" : df["Question"].dropna().astype(str).tolist(),
            "small_talk": small_talk_flat
        }