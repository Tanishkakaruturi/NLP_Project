import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
file_path = r"C:\Users\tanis\Downloads\tfidf_features.xlsx"  
df = pd.read_excel(file_path)

X = df.drop(columns=["label"]) 
y = df["label"]  
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

log_model = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs', multi_class='multinomial')
log_model.fit(X_train, y_train)

y_pred = log_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Logistic Regression Accuracy: {accuracy:.2f}") 
print("\nClassification Report:\n", classification_report(y_test, y_pred)) 
