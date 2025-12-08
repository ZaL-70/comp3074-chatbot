import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
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

# 5-fold stratified CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred = cross_val_predict(model, X, y, cv=cv)
scores = cross_val_score(model, X, y, cv=cv)

print("\n---- Intent Matcher Classification ----")
print("5-Fold CV Scores: ", scores)
print("\nClassification Report: \n", classification_report(y, y_pred))
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
# CUQ findings 16 questions x 5 participants
data = {
    'Q1':  [4,3,4,3,5],
    'Q2':  [2,3,2,4,1],
    'Q3':  [5,4,4,4,5],
    'Q4':  [1,2,2,3,1],
    'Q5':  [5,4,4,5,5],
    'Q6':  [1,2,2,1,1],
    'Q7':  [3,4,3,4,4],
    'Q8':  [3,3,4,2,2],
    'Q9':  [3,3,4,3,4],
    'Q10': [4,3,2,3,2],
    'Q11': [4,4,5,4,5],
    'Q12': [2,3,2,3,1],
    'Q13': [3,3,4,4,4],
    'Q14': [3,3,2,2,1],
    'Q15': [4,4,3,4,5],
    'Q16': [2,2,3,2,1]
}

df = pd.DataFrame(data)

# Reverse-score even-numbered questions (assuming 1-5 Likert scale)
even_questions = [f'Q{i}' for i in range(2, 17, 2)]
for q in even_questions:
    df[q] = 6 - df[q]  # Reverse: 1→5, 2→4, 3→3, 4→2, 5→1

# Convert DataFrame to long format
df_long = df.melt(var_name='Question', value_name='Response')

# Pivot into a frequency table for heatmap
response_pivot = df_long.pivot_table(
    index='Question',
    columns='Response',
    aggfunc=len,
    fill_value=0
)

# Order Q1–Q16 numerically
desired_order = [f"Q{i}" for i in range(1, 17)]
response_pivot = response_pivot.reindex(desired_order)

# Ensure all response values 1-5 are represented
for col in range(1, 6):
    if col not in response_pivot.columns:
        response_pivot[col] = 0

# Sort columns to ensure proper order
response_pivot = response_pivot[[1, 2, 3, 4, 5]]

# Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(response_pivot, annot=True, cmap="YlGnBu", fmt="d")

plt.title("CUQ Responses - Heatmap (Even Questions Reverse-Scored)", fontsize=16)
plt.xlabel("Likert score responses")
plt.ylabel("Question")
plt.tight_layout()
plt.show()