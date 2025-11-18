import joblib
import re

# Load the trained classifier
print('Loading offensive_logreg_classifier.joblib...')
classifier = joblib.load('offensive_logreg_classifier.joblib')
print('Classifier loaded successfully!\n')

def preprocess_text(text):
    """Preprocess text for classifier (same as training)."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Test messages
test_messages = [
    # Subtle/borderline offensive
    "is it pink?",
    "do you eat dogs",
    "are you a woman",
    "go to kitchen",
    "show feet",
    "add me on discord babe",
    "you should smile more",
    "women ☕",
    "make me a sandwich",
    "arent you supposed to be cooking",
    
    # Potentially offensive/harassment
    "lol what a noob",
    "you suck at this game",
    "uninstall",
    "trash gameplay",
    "are you boosted?",
    
    # Normal chat
    "hello hows your day",
    "nice play!",
    "what build are you using?",
    "poggers",
    "gg",
    "KEKW",
    "first time here, loving the vibes",
    
    # Emote spam (should be allowed)
    "LULW LULW LULW",
    "Pog Pog Pog",
    
    # Trauma dumping (should be allowed per instructions)
    "i was sick and my gf left me for a friend of mine even i got engaged with her. im depressed and i want to die.",
    "my dog died today im so sad",
    
    # Political (should be flagged)
    "trump 2024",
    "vote for biden",
    
    # Social media requests (should be flagged)
    "add me on instagram",
    "whats your snapchat",
    "follow me on twitter"
]

# Test each message
for msg in test_messages:
    print(f"Testing message: '{msg}'")
    print("-" * 50)
    
    # Preprocess the message
    clean_text = preprocess_text(msg)
    
    # Get prediction and probability
    prediction = classifier.predict([clean_text])[0]
    probabilities = classifier.predict_proba([clean_text])[0]
    
    # Assuming binary classification: 0 = not offensive, 1 = offensive
    is_offensive = prediction == 1
    confidence = probabilities[prediction]
    
    print(f"Preprocessed text: '{clean_text}'")
    print(f"Prediction: {'OFFENSIVE' if is_offensive else 'NOT OFFENSIVE'}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Probabilities: Not Offensive={probabilities[0]:.2%}, Offensive={probabilities[1]:.2%}")
    print("=" * 50)
    print()
