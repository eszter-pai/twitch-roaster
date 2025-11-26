from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load HateBERT model and tokenizer
print('Loading HateBERT model...')
model_name = "GroNLP/hateBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
print('HateBERT loaded successfully!\n')

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
