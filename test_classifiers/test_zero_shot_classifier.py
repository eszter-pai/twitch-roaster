from transformers import pipeline
import json
from datetime import datetime
from pathlib import Path

# Initialize zero-shot classifier
print('Loading zero-shot classifier...')
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
print('Classifier loaded successfully!\n')

# Define offensive categories
OFFENSIVE_CATEGORIES = [
    "chat message contains racism",
    "chat message contains sexism", 
    "chat message contains political opinion",
    "chat message contains insult",
    "chat message contains requests to add on social platforms"
]

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
    
    result = classifier(msg, OFFENSIVE_CATEGORIES, multi_label=True)
    
    # Check if any offensive category has high confidence (>0.5)
    max_score = max(result['scores'])
    is_likely_offensive = max_score > 0.5
    
    print(f"Classification results:")
    for label, score in zip(result['labels'], result['scores']):
        print(f"  {label}: {score:.2%}")
    print(f"\nMax score: {max_score:.2%}")
    print(f"Likely offensive (>0.5 threshold): {is_likely_offensive}")
    print("=" * 50)
    print()
    
    # Store result
    result_dict = {
        'text': msg,
        'is_likely_offensive': bool(is_likely_offensive),
        'max_score': float(max_score),
        'categories': {label: float(score) for label, score in zip(result['labels'], result['scores'])}
    }
    all_results.append(result_dict)

# Save results to JSON
output_data = {
    'model': 'facebook/bart-large-mnli',
    'classifier_type': 'zero-shot-classification',
    'timestamp': datetime.now().isoformat(),
    'threshold': 0.5,
    'total_messages': len(test_messages),
    'offensive_count': sum(1 for r in all_results if r['is_likely_offensive']),
    'not_offensive_count': sum(1 for r in all_results if not r['is_likely_offensive']),
    'offensive_categories': OFFENSIVE_CATEGORIES,
    'results': all_results
}

output_file = results_dir / 'zero_shot_results.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"\n✅ Results saved to {output_file}")
print(f"Summary: {output_data['offensive_count']}/{output_data['total_messages']} messages flagged as likely offensive")
