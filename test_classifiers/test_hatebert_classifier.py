from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
import os
from datetime import datetime
from pathlib import Path

# Load HateBERT model and tokenizer
print('Loading HateBERT model...')
model_name = "GroNLP/hateBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
print('HateBERT loaded successfully!\n')

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
results_dir = Path('test_classifiers/results')
results_dir.mkdir(exist_ok=True)

# Collect all results
all_results = []

# Test each message
for msg in test_messages:
    print(f"Testing message: '{msg}'")
    print("-" * 50)
    
    # Tokenize and prepare input
    inputs = tokenizer(msg, return_tensors="pt", truncation=True, max_length=512, padding=True)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1).item()
    
    # Extract probabilities
    prob_not_hate = probabilities[0][0].item()
    prob_hate = probabilities[0][1].item()
    
    is_hateful = prediction == 1
    confidence = prob_hate if is_hateful else prob_not_hate
    
    print(f"Prediction: {'HATEFUL/OFFENSIVE' if is_hateful else 'NOT HATEFUL'}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Probabilities: Not Hateful={prob_not_hate:.2%}, Hateful={prob_hate:.2%}")
    print("=" * 50)
    print()
    
    # Store result
    all_results.append({
        'text': msg,
        'is_hateful': bool(is_hateful),
        'confidence': float(confidence),
        'prob_not_hate': float(prob_not_hate),
        'prob_hate': float(prob_hate)
    })

# Save results to JSON
output_data = {
    'model': model_name,
    'timestamp': datetime.now().isoformat(),
    'total_messages': len(test_messages),
    'hateful_count': sum(1 for r in all_results if r['is_hateful']),
    'not_hateful_count': sum(1 for r in all_results if not r['is_hateful']),
    'results': all_results
}

output_file = results_dir / 'hatebert_results.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"\n✅ Results saved to {output_file}")
print(f"Summary: {output_data['hateful_count']}/{output_data['total_messages']} messages flagged as hateful")
