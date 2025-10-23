import pandas as pd
from textblob import TextBlob
file_path = r"C:\Users\tanis\Downloads\preprocessed_dataset.xlsx"  
df = pd.read_excel(file_path)
def get_sentiment(text):
    analysis = TextBlob(str(text))  # Convert to string to avoid errors
    polarity = analysis.sentiment.polarity  # Polarity score (-1 to 1)
    if polarity > 0:
        return "Positive"
    elif polarity == 0:
        return "Neutral"
    else:
        return "Negative"

df["sentiment"] = df["preprocessed_text"].apply(get_sentiment)
output_path = r"C:\Users\tanis\Downloads\sentiment_results.xlsx"
df.to_excel(output_path, index=False)
print(df[['preprocessed_text', 'sentiment']].head())

