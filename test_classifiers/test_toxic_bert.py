from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

# Load model and tokenizer
model_name = "unitary/toxic-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

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

# Get label mapping
id2label = model.config.id2label
print(f"Model: {model_name}")
print(f"Label mapping: {id2label}\n")
print("="*100)

# Test each message
all_results = []
for input_text in test_messages:
    # Tokenize and run inference
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).numpy()[0]
    
    # Create results dict
    result = {id2label[i]: probs[i] for i in range(len(probs))}
    result['text'] = input_text[:50] + ('...' if len(input_text) > 50 else '')
    all_results.append(result)

# Create DataFrame
df = pd.DataFrame(all_results)
# Move text column to front
cols = ['text'] + [col for col in df.columns if col != 'text']
df = df[cols]

print(df.round(5).to_string())
print("\n" + "="*100)
print("\nSummary of high toxicity scores (>0.5):")
high_toxicity_messages = []
for idx, row in df.iterrows():
    text = test_messages[idx]
    high_scores = []
    for label in id2label.values():
        if row[label] > 0.5:
            high_scores.append(f"{label}: {row[label]:.3f}")
    if high_scores:
        print(f"\n'{text[:60]}{'...' if len(text) > 60 else ''}')")
        for score in high_scores:
            print(f"  {score}")
        high_toxicity_messages.append(text)

# Save results to JSON
script_dir = Path(__file__).parent
results_dir = script_dir / 'results'
results_dir.mkdir(exist_ok=True)

# Prepare results for JSON
json_results = []
for idx, row in df.iterrows():
    result_dict = {
        'text': test_messages[idx],
        'scores': {label: float(row[label]) for label in id2label.values()}
    }
    json_results.append(result_dict)

output_data = {
    'model': model_name,
    'timestamp': datetime.now().isoformat(),
    'total_messages': len(test_messages),
    'high_toxicity_count': len(high_toxicity_messages),
    'label_mapping': id2label,
    'results': json_results
}

output_file = results_dir / 'toxic_bert_results.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"\n\n✅ Results saved to {output_file}")
print(f"Summary: {len(high_toxicity_messages)}/{len(test_messages)} messages with high toxicity scores (>0.5)")
