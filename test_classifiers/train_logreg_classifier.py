import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import re

def preprocess_text(text):
    """Clean and preprocess text data."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions and hashtags (but keep the text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def load_and_prepare_data(csv_path):
    """Load the labeled data and prepare it for training."""
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    # The dataset has a 'class' column where:
    # 0 = hate speech
    # 1 = offensive language
    # 2 = neither
    # We'll treat 0 and 1 as "offensive" (1) and 2 as "not offensive" (0)
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nClass distribution:")
    print(df['class'].value_counts())
    
    # Create binary labels: offensive (hate_speech or offensive_language) vs not offensive
    df['is_offensive'] = (df['class'] != 2).astype(int)
    
    print(f"\nBinary class distribution:")
    print(df['is_offensive'].value_counts())
    
    # Preprocess the tweets
    print("\nPreprocessing text...")
    df['clean_tweet'] = df['tweet'].apply(preprocess_text)
    
    return df

def train_classifier(X_train, y_train):
    """Train a text classification pipeline."""
    print("\nTraining classifier...")
    
    # Create a pipeline with TF-IDF vectorizer and Logistic Regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),  # Use unigrams and bigrams
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.8,  # Ignore terms that appear in more than 80% of documents
            strip_accents='unicode',
            lowercase=True,
            stop_words='english'
        )),
        ('classifier', LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'  # Handle class imbalance
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    
    return pipeline

def evaluate_classifier(pipeline, X_test, y_test):
    """Evaluate the trained classifier."""
    print("\nEvaluating classifier...")
    
    y_pred = pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Not Offensive', 'Offensive']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return accuracy

def main():
    # Load and prepare data
    df = load_and_prepare_data('data/labeled_data.csv')
    
    # Split the data
    X = df['clean_tweet']
    y = df['is_offensive']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train the classifier
    pipeline = train_classifier(X_train, y_train)
    
    # Evaluate the classifier
    accuracy = evaluate_classifier(pipeline, X_test, y_test)
    
    # Save the trained model
    model_path = 'offensive_classifier.joblib'
    print(f"\nSaving model to {model_path}...")
    joblib.dump(pipeline, model_path)
    print("Model saved successfully!")
    
    # Test with some example messages
    print("\n" + "="*50)
    print("Testing with example messages:")
    print("="*50)
    
    test_messages = [
        "Hey everyone, hope you're having a great day!",
        "You're such a stupid bitch",
        "This game is awesome",
        "Add me on Instagram babe",
        "Nice play!",
        "You suck at this game loser",
        "is it pink",
        "go to kitchen",
        "do you eat dogs"
    ]
    
    for msg in test_messages:
        clean_msg = preprocess_text(msg)
        prediction = pipeline.predict([clean_msg])[0]
        probability = pipeline.predict_proba([clean_msg])[0]
        
        print(f"\nMessage: {msg}")
        print(f"Prediction: {'OFFENSIVE' if prediction == 1 else 'NOT OFFENSIVE'}")
        print(f"Confidence: {probability[prediction]:.2%}")

if __name__ == "__main__":
    main()
