# classifies-customer-support-tickets-

Here is a comprehensive README.md file for your project. You can save this text as a file named README.md in your project directory.

# 🎫 Customer Support Ticket Classifier

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Gradio](https://img.shields.io/badge/UI-Gradio-purple)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end Machine Learning pipeline designed to automate the triage of customer support tickets. This system processes raw text to classify issues, assess urgency, and extract key entities, reducing manual workload for support teams.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Installation](#-installation)


---

## 📖 Project Overview

Handling a high volume of support tickets manually is slow and error-prone. This project leverages Natural Language Processing (NLP) to:
1.  **Classify Issue Types**: Automatically categories tickets (e.g., *Billing, Technical, Hardware*).
2.  **Predict Urgency**: Flags critical issues (e.g., *High, Medium, Low*).
3.  **Extract Entities**: Identifies products, dates, and order IDs for quick context.

---

## ✨ Key Features

* **Robust Preprocessing**: Includes text normalization, stopword removal, and lemmatization.
* **Dual-Model Classification**: Uses **Random Forest** classifiers optimized for multi-task learning.
* **Smart Feature Engineering**: Combines **TF-IDF** vectors with meta-features like *Ticket Length* and *Sentiment Analysis* (VADER).
* **Entity Extraction**: Hybrid rule-based system (Regex + Keywords) to find:
    * Products (e.g., "Laptop", "Router")
    * Dates (e.g., "2023-10-12", "yesterday")
    * Complaint Keywords (e.g., "broken", "fail")
* **Interactive Web UI**: Built with **Gradio** for real-time testing and demoing.

---

## ⚙️ Installation

1.  **Clone the Repository**
    ```bash
    [git clone [https://github.com/your-username/ticket-classifier.git](https://github.com/your-username/ticket-classifier.git)
    cd ticket-classifier](https://github.com/Geniusram/classifies-customer-support-tickets-.git)
    ```

2.  **Set up Virtual Environment (Optional but Recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install pandas numpy scikit-learn nltk gradio openpyxl scipy
    ```

4.  **Download NLTK Data**
    Run the included setup script or execute:
    ```python
    import nltk
    nltk.download(['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'vader_lexicon'])
    ```

---

## Run python code 
```bash
gradio_code.py
```
**or**
Double click gradio_code.py
---
## Run jupyter notebook
Run each cells
