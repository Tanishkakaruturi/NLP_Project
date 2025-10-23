import pandas as pd
import re
file_path = r"C:\Users\tanis\Downloads\project_dataset.xlsx" 
df = pd.read_excel(file_path)
basic_stopwords = {"the", "is", "in", "and", "to", "of", "a", "for", "on", "with", "at", "by", "from", "it", "this", "that"}
def fast_preprocess_text(text):
    if isinstance(text, str):
        text = re.sub(r'http\S+|www\S+', '', text)  
        text = re.sub(r'@\w+|#\w+', '', text)  
        text = re.sub(r'\d+', '', text)  
        text = re.sub(r'[^\w\s]', '', text)  
        text = text.lower().strip()  
        tokens = text.split()
        tokens = [word for word in tokens if word not in basic_stopwords]
        return " ".join(tokens)
    return ""

df['preprocessed_text'] = df['text'].apply(fast_preprocess_text)
output_path = r"C:\Users\tanis\Downloads\preprocessed_dataset.xlsx"
df.to_excel(output_path, index=False)

print("Preprocessing complete! Cleaned dataset saved at:", output_path)
print(df[['text', 'preprocessed_text']].head()) 


