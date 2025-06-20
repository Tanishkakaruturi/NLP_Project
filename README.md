Sentiment Analysis of HMPV Tweets

This project analyzes public sentiment toward HMPV virus based on tweets collected during its recent outbreak. Using natural language processing and deep learning models, we classify tweets into positive, negative, or neutral categories to better understand public perception, awareness, and emotional response during the first 30 days of the outbreak.

Project Overview
Objective: To analyze public sentiment around HMPV by classifying tweet texts into sentiment categories.
Data Source: Tweets collected using the Twitter web scraper.
Techniques Used:
  - Data preprocessing (cleaning, tokenization)
  - Sentiment classification using TextBlob
  - Feature extraction
  - Sentiment classification using ML models

---

Dataset

- Size: 9,700+ tweets
- Attributes:
  - `user_name`, `user_location`, `user_followers`, `user_verified`, `text`, `hashtags`, `retweets`, `likes`, `date`, `source`, `is_retweeted`


Tools & Technologies

- Python 
- NLTK, SpaCy, Transformers (Hugging Face)
- Scikit-learn
- Jupyter Notebook
- Twitter API (Tweepy)

--Preprocessing Steps

- Removal of URLs, mentions, special characters
- Lowercasing and tokenization
- Stopword removal
- Lemmatization
- Hashtag and emoji handling

Results

| Model            | Accuracy |
|------------------|----------|
| Random Classifier| 75%      |
| Naive Bayes      | 60%      |
| SVM              | 83%      |



Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC Curve




