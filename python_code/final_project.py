# 1.Assignment Tasks

import pandas as pd
import numpy as np
import re

# A. Load Data

data = pd.read_excel("dataset1.xls")

df = pd.DataFrame(data)
print(df.head())

# B. Handling Missing Data

# We drop rows where the primary feature 'ticket_text' is missing
df_clean = df.dropna().copy()

# C. Preprocessing Functions


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


# Initialize Lemmatizer and Stopwords list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def clean_and_preprocess(text):
    # 1. Text Normalization: Convert to lowercase
    text = str(text).lower()

    # 2. Remove special characters and numbers
    # Keeping only alphabets and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # 3. Tokenization
    tokens = text.split()

    # 4. Stopword Removal and Lemmatization
    processed_tokens = [
        lemmatizer.lemmatize(word) 
        for word in tokens 
        if word not in stop_words and len(word) > 2  # Removing very short words
    ]

    return " ".join(processed_tokens)

    
# Apply the preprocessing
df_clean['processed_text'] = df_clean['ticket_text'].apply(clean_and_preprocess)


# D. Output Verification

print("Original Rows:", len(df))
print("Cleaned Rows:", len(df_clean))
print("\nSample Data:")
print(df_clean[['ticket_text', 'processed_text']].head())


# 2. Feature Engineering

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.sparse import hstack
nltk.download('vader_lexicon')

# A. Text Vectorization (TF-IDF)


# We limit max_features to keep the dataset manageable (e.g., top 1000 words)
tfidf_vectorizer = TfidfVectorizer(max_features=1000)

# Fit and transform the processed text
tfidf_matrix = tfidf_vectorizer.fit_transform(df_clean['processed_text'])

# Convert to DataFrame for visualization (optional, usually kept as sparse matrix)
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

print(f"TF-IDF Matrix Shape: {tfidf_df.shape}")


# B. Extract Additional Features

sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    # Returns a compound score between -1 (Negative) and +1 (Positive)
    return sia.polarity_scores(str(text))['compound']

# Feature 1: Ticket Length (Character count of original text)
# Rationale: Technical logs or angry rants might be significantly longer.
df_clean['ticket_length'] = df_clean['ticket_text'].apply(len)

# Feature 2: Sentiment Score
# Rationale: 'High' urgency tickets often contain negative sentiment words (angry, fail, bad).
df_clean['sentiment_score'] = df_clean['ticket_text'].apply(get_sentiment)


# C. Combine Features


# Combine metadata features with TF-IDF vectors
# Note: For many ML models, you can stack these horizontally.
meta_features = df_clean[['ticket_length', 'sentiment_score']].values

print("\n--- Feature Engineering Results ---")
print(df_clean[['ticket_text', 'ticket_length', 'sentiment_score']].head())

# Combine with TF-IDF Matrix
# hstack (horizontal stack) appends the metadata columns to the right side of the TF-IDF matrix
X_combined = hstack([tfidf_matrix, meta_features])

# Verification
print(f"Shape of TF-IDF Matrix: {tfidf_matrix.shape}")
print(f"Shape of Metadata:      {meta_features.shape}")
print(f"Shape of Combined Data: {X_combined.shape}")


# 3. Multi-Task Learning

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# A. Prepare Data Splits

# Define our target variables
y_issue = df_clean['issue_type']
y_urgency = df_clean['urgency_level']

# Split the data into Training and Testing sets
# We use the same random_state to ensure X_test corresponds to both y_issue_test and y_urgency_test
X_train, X_test, y_issue_train, y_issue_test, y_urgency_train, y_urgency_test = train_test_split(
    X_combined, 
    y_issue, 
    y_urgency, 
    test_size=0.25, 
    random_state=42
)

print(f"Training Data Shape: {X_train.shape}")
print(f"Testing Data Shape:  {X_test.shape}")


# B. Model 1: Issue Type Classifier

print("\n--- Training Issue Type Classifier ---")
clf_issue = RandomForestClassifier(n_estimators=100, random_state=42)
clf_issue.fit(X_train, y_issue_train)

# Predictions
y_issue_pred = clf_issue.predict(X_test)

# Evaluation
print("Issue Type Classification Report:")
print(classification_report(y_issue_test, y_issue_pred))


# C. Model 2: Urgency Level Classifier

print("\n--- Training Urgency Level Classifier ---")
# Using class_weight='balanced' to handle potential imbalance (e.g., fewer 'Critical' tickets)
clf_urgency = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
clf_urgency.fit(X_train, y_urgency_train)

# Predictions
y_urgency_pred = clf_urgency.predict(X_test)

# Evaluation
print("Urgency Level Classification Report:")
print(classification_report(y_urgency_test, y_urgency_pred))


# 4. Entity Extraction

# A. Define Knowledge Bases

products = df_clean['product'].str.lower().unique().tolist()
complaint_keywords = ['broken', 'late', 'error', 'fail', 'crash', 'damaged', 'flickering', 'stuck', 'slow','malfunction','issue','problem','not working','stopped working','unresponsive','delay','missing','incorrect','unable','disconnect','overheated','repair','replace','refund','warranty','faulty','defective','glitch','hang','freeze','lag','unusable','corrupt','virus','spyware','adware','phishing','scam','spam','hijack','breach','intrusion','theft','loss','payment issue','instead','insufficient','unauthorized','billing','charge','invoice','subscription']


# B. Define Extraction Function

def extract_entities(text):
    text_lower = str(text).lower()
    entities = {
        'products': [],
        'dates': [],
        'complaint_keywords': []
    }

    # --- A. Extract Products (Keyword Matching) ---
    # Check if any known product appears in the text
    for product in products:
        # Use regex boundary \b to avoid partial matches (e.g., preventing "net" matching inside "internet")
        if re.search(r'\b' + re.escape(product) + r'\b', text_lower):
            entities['products'].append(product)

    # --- B. Extract Complaint Keywords ---
    for keyword in complaint_keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
            entities['complaint_keywords'].append(keyword)

    # --- C. Extract Dates (Regex Patterns) ---
    # Pattern 1: YYYY-MM-DD or DD-MM-YYYY or similar (e.g., 2023-10-05, 10/05/2023)
    date_pattern_num = r'\b\d{1,4}[-/]\d{1,2}[-/]\d{2,4}\b'

    # Pattern 2: Textual dates (e.g., "Jan 5th", "5 October")
    date_pattern_text = r'(?:\d{1,2}(?:st|nd|rd|th)?\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s+\d{4})?'

    # Find all matches
    matches_num = re.findall(date_pattern_num, text)
    matches_text = re.findall(date_pattern_text, text, re.IGNORECASE)

    entities['dates'] = matches_num + matches_text

    return entities


# C. Apply to DataFrame

# We apply this to the original 'ticket_text' to preserve formatting/casing for date extraction if needed
df_clean['extracted_entities'] = df_clean['ticket_text'].apply(extract_entities)


# D. Display Results

for index, row in df_clean.head().iterrows():
    print(f"Ticket: {row['ticket_text']}")
    print(f"Entities: {row['extracted_entities']}")
    print("-" * 30)


# 5. Integration

def process_ticket(raw_text):
    """
    End-to-end pipeline for a single support ticket.

    Args:
        raw_text (str): The raw customer support ticket text.

    Returns:
        dict: A dictionary containing predictions and extracted entities.
    """
    # -------------------------------------------------------
    # 1. Preprocessing
    # -------------------------------------------------------
    # Use the cleaning function defined in Step 1
    cleaned_text = clean_and_preprocess(raw_text)

    # -------------------------------------------------------
    # 2. Feature Engineering
    # -------------------------------------------------------
    # A. TF-IDF Vectorization
    # Note: We wrap cleaned_text in a list [] because transform expects an iterable
    tfidf_vec = tfidf_vectorizer.transform([cleaned_text])

    # B. Metadata Extraction
    ticket_len = len(raw_text)
    # Use the Sentiment Analyzer defined in Step 2
    sentiment = sia.polarity_scores(str(raw_text))['compound']

    # Create a 2D array for metadata [[len, sentiment]]
    meta_features = np.array([[ticket_len, sentiment]])

    # C. Combine Features
    # Stack the sparse TF-IDF vector with the dense metadata
    X_input = hstack([tfidf_vec, meta_features])

    # -------------------------------------------------------
    # 3. Model Prediction
    # -------------------------------------------------------
    # [0] is used to get the string value from the prediction array
    predicted_issue = clf_issue.predict(X_input)[0]
    predicted_urgency = clf_urgency.predict(X_input)[0]

    # -------------------------------------------------------
    # 4. Entity Extraction
    # -------------------------------------------------------
    # Use the extraction function defined in Step 4
    entities = extract_entities(raw_text)

    # -------------------------------------------------------
    # 5. Construct Output
    # -------------------------------------------------------
    result = {
        "raw_text": raw_text,
        "predicted_issue_type": predicted_issue,
        "predicted_urgency_level": predicted_urgency,
        "extracted_entities": entities
    }

    return result


# Test with a new, unseen ticket
new_ticket = "Vision LED TV is no response. It stopped working after just 9 days"


result = process_ticket(new_ticket)

# Pretty print the result
import json
print(json.dumps(result, indent=4))

