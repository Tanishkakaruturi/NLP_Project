import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the TF-IDF feature dataset
file_path = r"C:\Users\tanis\Downloads\tfidf_features.xlsx"  # Update the path if needed
df = pd.read_excel(file_path)

# Separate features (X) and sentiment labels (y)
X = df.drop(columns=["label"])  # TF-IDF Features
y = df["label"]  # Sentiment Labels (0 = Negative, 1 = Neutral, 2 = Positive)

# Split into training (80%) and testing (20%) datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Naïve Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = nb_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display results
print(f"Naïve Bayes Accuracy: {accuracy:.2f}")  # Display accuracy
print("\nClassification Report:\n", classification_report(y_test, y_pred))  # Detailed metrics






