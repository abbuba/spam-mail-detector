import nltk
import os
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# --- START: MANUAL NLTK PATH FIX ---
# This block forces NLTK to look for data in your user's home directory.
try:
    # Define the path to the nltk_data directory
    nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
    
    # Check if this path is already in NLTK's search paths
    if nltk_data_dir not in nltk.data.path:
        # If not, add it to the search paths
        nltk.data.path.append(nltk_data_dir)
        print(f"Manually added NLTK data path: {nltk_data_dir}")

    # Now, try a function that requires a downloaded package to test it
    _ = stopwords.words('english')
    print("NLTK data path configured successfully.\n")

except LookupError:
    # This is a fallback if the manual path also fails
    print("NLTK data not found even after manual path setting. Attempting download...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
# --- END: MANUAL NLTK PATH FIX ---


# --- 1. Load the Dataset ---
print("Step 1: Loading the dataset...")
filepath = "SMSSpamCollection" 
df = pd.read_csv(filepath, sep='\t', header=None, names=['label', 'message'])
print("Dataset loaded successfully.\n")


# --- 2. Preprocess the Text Data ---
print("Step 2: Preprocessing text data...")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'[^a-z]', ' ', text.lower())
    tokens = nltk.word_tokenize(text)
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(processed_tokens)

df['processed_message'] = df['message'].apply(preprocess_text)
print("Text preprocessing complete.\n")


# --- 3. Convert Text to Numeric Features (TF-IDF) ---
print("Step 3: Converting text to numeric features using TF-IDF...")
tfidf_vectorizer = TfidfVectorizer(max_features=3000)
X = tfidf_vectorizer.fit_transform(df['processed_message']).toarray()
y = df['label'].map({'spam': 1, 'ham': 0})
print("TF-IDF vectorization complete.\n")


# --- 4. Split Data and Train a Model ---
print("Step 4: Splitting data and training the Naive Bayes model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)
print("Model training complete.\n")


# --- 5. Evaluate the Model ---
print("Step 5: Evaluating the model performance...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("--- RESULTS ---")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
print("---------------")
