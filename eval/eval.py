import numpy as np
import pandas as pd
from chatbot.QAModule import *

# ---- Evaluate the QA system ----
train_qa_set = pd.read_csv("../data/cooking-qa.csv")
qa = QAModule(train_qa_set)

# Load the test data for evaluation
test_qa_set = pd.read_csv("test_data/test-qa.csv")

correct_count = 0
no_answer_count = 0
incorrect_count = 0

for index, item in test_qa_set.iterrows():
    system_answer = qa.get_answer(item["Question"])

    if system_answer == item["Answer"]:
        correct_count += 1
    elif system_answer == "no_answer":
        no_answer_count += 1
    else:
        incorrect_count += 1

# Calculate and print accuracy
total_questions = len(test_qa_set)
accuracy = correct_count / total_questions
print(f"Accuracy: {accuracy:.2%}")
print(f"\nBreakdown:")
print(f"- Incorrect answers: {incorrect_count}/{total_questions}")
print(f"- Correct answers: {correct_count}/{total_questions}")
print(f"- No answer found: {no_answer_count}/{total_questions}")