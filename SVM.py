import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
file_path = r"C:\Users\tanis\Downloads\tfidf_features.xlsx"
df = pd.read_excel(file_path)


X = df.drop(columns=["label"]) 
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

svm_model = SVC(kernel='linear', probability=True, random_state=42)  
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
y_proba = svm_model.predict_proba(X_test)  # Needed for ROC-AUC

accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("Precision (Macro):", precision_score(y_test, y_pred, average='macro'))
print("Recall (Macro):", recall_score(y_test, y_pred, average='macro'))
print("F1 Score (Macro):", f1_score(y_test, y_pred, average='macro'))
print("Precision (Weighted):", precision_score(y_test, y_pred, average='weighted'))
print("Recall (Weighted):", recall_score(y_test, y_pred, average='weighted'))
print("F1 Score (Weighted):", f1_score(y_test, y_pred, average='weighted'))

# Confusion Matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))



