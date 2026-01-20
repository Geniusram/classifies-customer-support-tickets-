# classifies-customer-support-tickets-

Here is a comprehensive README.md file for your project. You can save this text as a file named README.md in your project directory.

🎫 Intelligent Customer Support Ticket Classifier
📖 Project Description
This project implements an end-to-end Machine Learning pipeline designed to automate the initial handling of customer support tickets. By leveraging Natural Language Processing (NLP) and Machine Learning techniques, the system analyzes raw ticket text to:

Classify the Issue Type (e.g., Technical, Billing, Product).

Predict the Urgency Level (e.g., Critical, High, Medium, Low).

Extract Key Entities (Product names, dates, complaint keywords, and Order IDs).

This solution aims to reduce manual triage time and ensure critical issues are flagged immediately.

Features Implemented
Data Preprocessing Pipeline:

Text normalization (lowercase, noise removal).

Tokenization and Stopword removal.

Lemmatization using NLTK to standardize vocabulary.

Handling of missing values.

Feature Engineering:

TF-IDF Vectorization: Converts text into meaningful numerical features.

Meta-Features: Calculates "Ticket Length" and "Sentiment Score" (using VADER) to improve urgency detection.

Multi-Task Classification:

Two separate Random Forest Classifiers trained on a shared feature set to predict Issue Type and Urgency simultaneously.

Handles class imbalance using weighted parameters.

Entity Extraction:

Hybrid approach using Regex (for dates and IDs) and Keyword Matching (for products and complaint types).

Interactive UI:

A Gradio web interface allows users to test the model in real-time.

Installation
Prerequisites
Python 3.8 or higher

pip (Python package manager)

1. Clone the Repository
Bash

git clone https://github.com/yourusername/ticket-classifier.git
cd ticket-classifier
2. Install Dependencies
Install the required Python libraries:

Bash

pip install pandas numpy scikit-learn nltk gradio openpyxl scipy
3. Download NLTK Data
The project requires specific NLTK corpora. You can download them by running this Python command once:

Python

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')
