import joblib
import re
import json
import os
from datetime import datetime
from pathlib import Path

# Load the trained classifier
print('Loading offensive_logreg_classifier.joblib...')
# Get the parent directory (project root) where the model file is located
script_dir = Path(__file__).parent
model_path = script_dir.parent / 'offensive_logreg_classifier.joblib'
classifier = joblib.load(model_path)
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

# Load test messages from file
def load_test_messages(filename='test_messages.txt'):
    """Load test messages from file, excluding comments."""
    messages = []
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    file_path = script_dir / filename
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith('#'):
                messages.append(line)
    return messages

test_messages = load_test_messages()

# Create results directory if it doesn't exist
script_dir = Path(__file__).parent
results_dir = script_dir / 'results'
results_dir.mkdir(exist_ok=True)

# Collect all results
all_results = []

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
    
    # Store result
    all_results.append({
        'text': msg,
        'preprocessed_text': clean_text,
        'is_offensive': bool(is_offensive),
        'confidence': float(confidence),
        'prob_not_offensive': float(probabilities[0]),
        'prob_offensive': float(probabilities[1])
    })

# Save results to JSON
output_data = {
    'model': 'Logistic Regression Classifier',
    'model_file': 'offensive_logreg_classifier.joblib',
    'timestamp': datetime.now().isoformat(),
    'total_messages': len(test_messages),
    'offensive_count': sum(1 for r in all_results if r['is_offensive']),
    'not_offensive_count': sum(1 for r in all_results if not r['is_offensive']),
    'results': all_results
}

output_file = results_dir / 'logreg_results.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"\n✅ Results saved to {output_file}")
print(f"Summary: {output_data['offensive_count']}/{output_data['total_messages']} messages flagged as offensive")
