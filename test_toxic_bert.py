from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

# Load model and tokenizer
model_name = "unitary/toxic-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

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
for idx, row in df.iterrows():
    text = test_messages[idx]
    high_scores = []
    for label in id2label.values():
        if row[label] > 0.5:
            high_scores.append(f"{label}: {row[label]:.3f}")
    if high_scores:
        print(f"\n'{text[:60]}{'...' if len(text) > 60 else ''}'")
        for score in high_scores:
            print(f"  {score}")
