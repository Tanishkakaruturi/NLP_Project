import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

file_path = r"C:\Users\tanis\Downloads\tfidf_features.xlsx" 
df = pd.read_excel(file_path)

X = df.drop(columns=["label"])  
y = df["label"]  # Sentiment Labels (0 = Negative, 1 = Neutral, 2 = Positive)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Naïve Bayes Accuracy: {accuracy:.2f}")  
print("\nClassification Report:\n", classification_report(y_test, y_pred))  






