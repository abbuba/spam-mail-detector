# Spam Mail Detector 📧

## Overview

This project is a machine learning-based **Spam Mail Detector** that classifies SMS messages as either "spam" or "ham" (not spam). It uses Natural Language Processing (NLP) techniques to process the text data and a **Multinomial Naive Bayes** classifier to make predictions.

This project was built to demonstrate skills in data preprocessing, feature engineering, and model building for a classic text classification problem.

---

##  Technologies Used

- **Language:** Python 3.x
- **Libraries:**
  - `pandas` for data manipulation
  - `scikit-learn` for machine learning (TF-IDF, model training, evaluation)
  - `nltk` for natural language processing (tokenization, stopwords, lemmatization)

---

##  Dataset

The project uses the [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) from the UCI Machine Learning Repository. It contains 5,572 SMS messages in English, tagged as ham or spam.

---

##  How to Run the Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/abbuba/spam-mail-detector.git](https://github.com/abbuba/spam-mail-detector.git)
    cd spam-mail-detector
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate

    # For Windows
    python -m venv .venv
    .venv\Scripts\activate
    ```

3.  **Install the required libraries:**
    (You should create a `requirements.txt` file by running `pip freeze > requirements.txt` in your activated environment).
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the script:**
    ```bash
    python spam_detector.py
    ```
The script will load the local data, train the model, and print the final evaluation report.

---

## Results

The model was evaluated on a test set (20% of the data) and achieved the following performance:

- **Accuracy:** **98.21%**

**Classification Report:**
precision    recall  f1-score   support

         Ham       0.98      1.00      0.99       966
        Spam       0.98      0.88      0.93       149

    accuracy                           0.98      1115
   macro avg       0.98      0.94      0.96      1115
weighted avg       0.98      0.98      0.98      1115

---------------
This shows that the model is highly effective at identifying legitimate messages (100% recall for Ham) and catches 88% of spam messages.
