import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from chatbot.IntentMatcher import IntentClassifier
from chatbot.QAModule import *

# ---- Evaluate the QA Module ----
train_qa_df = pd.read_csv("../data/cooking-qa.csv")
qa = QAModule(train_qa_df)

# Load the test data for evaluation
test_qa_df = pd.read_csv("test_data/test-qa.csv")

correct_count = 0
no_answer_count = 0
incorrect_count = 0

for index, item in test_qa_df.iterrows():
    system_answer = qa.get_answer(item["Question"])

    if system_answer == item["Answer"]:
        correct_count += 1
    elif system_answer == "no_answer":
        no_answer_count += 1
    else:
        incorrect_count += 1

# Calculate and print accuracy
total_questions = len(test_qa_df)
accuracy = correct_count / total_questions
print("\n---- QA EVALUATION ----")
print(f"Overall Accuracy: {accuracy:.2%}")
print(f"\nBreakdown:")
print(f"- Incorrect answers: {incorrect_count}/{total_questions}")
print(f"- Correct answers: {correct_count}/{total_questions}")
print(f"- No answer found: {no_answer_count}/{total_questions}")

# ---- Evaluate the Intent Matcher ----
intent_training_df = pd.read_csv("../data/intent-training.csv")

X = intent_training_df['text']
y = intent_training_df['intent']
class_labels = sorted(intent_training_df['intent'].unique())

model = IntentClassifier(intent_training_df).model

# 8-fold stratified CV (same as your debug)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred = cross_val_predict(model, X, y, cv=cv)
scores = cross_val_score(model, X, y, cv=cv)

print("\n---- Intent Matcher Classification ----")
print("5-Fold CV Scores: ", scores)
print("\nClassifier Report:\n", classification_report(y, y_pred))
cm = confusion_matrix(y, y_pred, labels=class_labels)

plt.figure(figsize=(12, 10))
sns.heatmap(cm,
            annot=True,
            cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels
            )

plt.title("Confusion Matrix for Intent Classifier (5-Fold CV)", fontsize=16)
plt.xlabel("Predicted Label", fontsize=14)
plt.ylabel("True Label", fontsize=14)

plt.tight_layout()
plt.show()

# ---- CUQ Results ----
