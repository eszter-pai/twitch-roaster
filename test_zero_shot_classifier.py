from transformers import pipeline

# Initialize zero-shot classifier
print('Loading zero-shot classifier...')
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
print('Classifier loaded successfully!\n')

# Define offensive categories
OFFENSIVE_CATEGORIES = [
    "racism or racist remarks",
    "sexism or sexist remarks", 
    "political discussions",
    "harassment or targeted attacks",
    "requests to add on social platforms"
]

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
