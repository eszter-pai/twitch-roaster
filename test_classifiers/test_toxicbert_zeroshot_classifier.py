from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
from datetime import datetime

# Initialize zero-shot classifier
print('Loading zero-shot classifier...')
zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
print('Zero-shot classifier loaded!')

# Load toxic-bert
print('Loading toxic-bert...')
toxic_model_name = "unitary/toxic-bert"
toxic_tokenizer = AutoTokenizer.from_pretrained(toxic_model_name)
toxic_model = AutoModelForSequenceClassification.from_pretrained(toxic_model_name)
toxic_model.eval()
print('Toxic-bert loaded!\n')

# Define offensive categories for zero-shot
OFFENSIVE_CATEGORIES = [
    "chat message contains racism",
    "chat message contains sexism", 
    "chat message contains political opinion",
    "chat message contains insult",
    "chat message contains requests to add on social platforms"
]

def get_toxic_bert_score(text):
    """Get the maximum toxicity score from toxic-bert."""
    inputs = toxic_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = toxic_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).numpy()[0]
    
    # Return max probability (excluding the 'toxic' label at index 0, or include it)
    # toxic-bert labels: 0=toxic, 1=severe_toxic, 2=obscene, 3=threat, 4=insult, 5=identity_hate
    max_score = max(probs)
    max_label = toxic_model.config.id2label[probs.argmax()]
    
    return max_score, max_label

def get_zero_shot_score(text):
    """Get the maximum confidence score from zero-shot classifier."""
    result = zero_shot_classifier(text, OFFENSIVE_CATEGORIES, multi_label=True)
    max_score = max(result['scores'])
    max_label = result['labels'][result['scores'].index(max_score)]
    
    return max_score, max_label

def classify_message(text, threshold=0.85):
    """
    Combine both classifiers and use the maximum confidence score.
    
    Args:
        text: Message to classify
        threshold: Confidence threshold (default 0.85 = 85%)
    
    Returns:
        dict with classification results
    """
    # Get scores from both models
    toxic_score, toxic_label = get_toxic_bert_score(text)
    zero_shot_score, zero_shot_label = get_zero_shot_score(text)
    
    # Use the maximum score
    if toxic_score >= zero_shot_score:
        max_score = toxic_score
        max_label = toxic_label
        model_used = "toxic-bert"
    else:
        max_score = zero_shot_score
        max_label = zero_shot_label
        model_used = "zero-shot"
    
    # Determine if toxic based on threshold
    is_toxic = max_score >= threshold
    
    return {
        'text': text,
        'is_toxic': bool(is_toxic),  # Convert to native Python bool
        'max_score': float(max_score),  # Convert to native Python float
        'max_label': max_label,
        'model_used': model_used,
        'toxic_bert_score': float(toxic_score),  # Convert to native Python float
        'toxic_bert_label': toxic_label,
        'zero_shot_score': float(zero_shot_score),  # Convert to native Python float
        'zero_shot_label': zero_shot_label
    }

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
    "SabaPing",
    "DinoDance",
    
    # Trauma dumping (should be allowed per instructions)
    "i was sick and my gf left me for a friend of mine even i got engaged with her. im depressed and i want to die.",
    "my dog died today im so sad",
    
    # Political (should be flagged)
    "trump 2024",
    "vote for biden",
    
    # Social media requests (should be flagged)
    "add me on instagram",
    "whats your snapchat",
    "follow me on twitter",

    # Others
    "Go touch Reginald",
    "I think we need to make him way less sensitive. Only to clap back if its really certain.",
    "Okay, but this is valid - tropes of racism and stuff in the Witcher can inspire valuable discussion"
]

print("=" * 100)
print("COMBINED CLASSIFIER TESTING (Threshold: 85%)")
print("=" * 100)

# Test with different thresholds
for threshold in [0.85, 0.75, 0.65]:
    print(f"\n\n{'='*100}")
    print(f"TESTING WITH THRESHOLD: {threshold:.0%}")
    print(f"{'='*100}\n")
    
    toxic_count = 0
    results_list = []
    
    for msg in test_messages:
        result = classify_message(msg, threshold=threshold)
        
        print(f"Message: '{msg[:70]}{'...' if len(msg) > 70 else ''}'")
        print(f"  Toxic-BERT:  {result['toxic_bert_score']:.2%} ({result['toxic_bert_label']})")
        print(f"  Zero-Shot:   {result['zero_shot_score']:.2%} ({result['zero_shot_label']})")
        print(f"  → MAX SCORE: {result['max_score']:.2%} (from {result['model_used']})")
        print(f"  → DECISION:  {'🚨 TOXIC' if result['is_toxic'] else '✅ OK'}")
        
        if result['is_toxic']:
            toxic_count += 1
        
        # Add result to list for JSON export
        results_list.append(result)
        
        print()
    
    print(f"\nSummary: {toxic_count}/{len(test_messages)} messages flagged as toxic at {threshold:.0%} threshold")
    
    # Save results to JSON file
    output_data = {
        'threshold': threshold,
        'threshold_percentage': f"{threshold:.0%}",
        'total_messages': len(test_messages),
        'toxic_count': toxic_count,
        'ok_count': len(test_messages) - toxic_count,
        'timestamp': datetime.now().isoformat(),
        'results': results_list
    }
    
    filename = f"classifier_results_threshold_{int(threshold*100)}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Results saved to {filename}")

print("\n" + "=" * 100)
print("Testing complete!")
