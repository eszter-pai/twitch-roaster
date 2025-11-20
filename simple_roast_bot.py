import re
import os
import json
from pathlib import Path
from dotenv import load_dotenv
import requests
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from twitchAPI.twitch import Twitch
from twitchAPI.oauth import UserAuthenticator, refresh_access_token
from twitchAPI.type import AuthScope, ChatEvent
from twitchAPI.chat import Chat, EventData, ChatMessage, ChatSub, ChatCommand
import asyncio
from collections import deque, defaultdict
from openai import OpenAI
from datetime import datetime, timedelta

load_dotenv()

BOT_NAME = os.getenv('TWITCH_BOT_USERNAME')
MODEL_URL = os.getenv('MODEL_API_URL')  # Ollama URL (kept for fallback)
APP_ID  = os.getenv('CLIENT_ID')
APP_SECRET  = os.getenv('CLIENT_SECRET')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
USER_SCOPE = [AuthScope.CHAT_READ, AuthScope.CHAT_EDIT]
TARGET_CHANNEL = os.getenv('TWITCH_CHANNEL')
EMOTE_USER_ID = os.getenv('SEVENTV_USER_ID')
TOKEN_FILE = 'twitch_tokens.json'

# Classifier settings
USE_CLASSIFIER = True
CLASSIFIER_THRESHOLD = 0.88
# CLASSIFIER_MODEL = 'offensive_classifier.joblib' # this is the trained logreg model


# Load the trained classifier
# print('Loading offensive message classifier...')
# classifier = joblib.load(CLASSIFIER_MODEL)
# print('Classifier loaded successfully!')

# Initialize DeepSeek client
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# Classifier components (loaded conditionally)
zero_shot_classifier = None
toxic_tokenizer = None
toxic_model = None

# Define offensive categories for zero-shot
OFFENSIVE_CATEGORIES = [
    "chat message contains racism",
    "chat message contains sexism", 
    "chat message contains political opinion",
    "chat message contains insult",
    "chat message contains requests to add on social platforms"
]

# Message history storage
general_chat_history = deque(maxlen=10)  # Last 10 messages from all users
user_message_history = defaultdict(lambda: deque(maxlen=3))  # Last 3 messages per user
user_called_out = defaultdict(bool)  # Track if user was recently called out

# 7TV Emote cache
emote_list_cache = None
emote_cache_time = None
EMOTE_CACHE_DURATION = timedelta(seconds=10)  # Refresh every 10 seconds (for testing)
emote_context = ""  # Global variable to store emote context

# Global emote lists for stripping
all_emote_names = set()  # Combined set of all emote names for stripping
emote_names_cache_time = None

def load_classifier_models():
    """Load the combined classifier models (toxic-bert + zero-shot)."""
    global zero_shot_classifier, toxic_tokenizer, toxic_model
    
    if not USE_CLASSIFIER:
        print('Classifier disabled (USE_CLASSIFIER=false)')
        return
    
    try:
        print('Loading zero-shot classifier...')
        zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        print('Zero-shot classifier loaded!')
        
        print('Loading toxic-bert...')
        toxic_model_name = "unitary/toxic-bert"
        toxic_tokenizer = AutoTokenizer.from_pretrained(toxic_model_name)
        toxic_model = AutoModelForSequenceClassification.from_pretrained(toxic_model_name)
        toxic_model.eval()
        print(f'Toxic-bert loaded! (threshold: {CLASSIFIER_THRESHOLD:.0%})')
    except Exception as e:
        print(f'Error loading classifier models: {e}')
        print('Classifier will be disabled for this session.')

def get_toxic_bert_score(text):
    """Get the maximum toxicity score from toxic-bert."""
    if toxic_tokenizer is None or toxic_model is None:
        return 0.0, "model_not_loaded"
    
    inputs = toxic_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = toxic_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).numpy()[0]
    
    max_score = max(probs)
    max_label = toxic_model.config.id2label[probs.argmax()]
    
    return max_score, max_label

def get_zero_shot_score(text):
    """Get the maximum confidence score from zero-shot classifier."""
    if zero_shot_classifier is None:
        return 0.0, "model_not_loaded"
    
    result = zero_shot_classifier(text, OFFENSIVE_CATEGORIES, multi_label=True)
    max_score = max(result['scores'])
    max_label = result['labels'][result['scores'].index(max_score)]
    
    return max_score, max_label

def classify_message(text, threshold=None):
    """Combine both classifiers and use the maximum confidence score."""
    if threshold is None:
        threshold = CLASSIFIER_THRESHOLD
    
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
        'is_toxic': bool(is_toxic),
        'max_score': float(max_score),
        'max_label': max_label,
        'model_used': model_used,
        'toxic_bert_score': float(toxic_score),
        'toxic_bert_label': toxic_label,
        'zero_shot_score': float(zero_shot_score),
        'zero_shot_label': zero_shot_label
    }

def fetch_7tv_emotes():
    """Fetch emotes from 7TV GraphQL API and format them for the LLM."""
    global emote_list_cache, emote_cache_time
    
    # Check if cache is still valid
    if emote_list_cache and emote_cache_time:
        if datetime.now() - emote_cache_time < EMOTE_CACHE_DURATION:
            return emote_list_cache

    try:
        # Fetch from 7TV GraphQL API
        query = """
        query GetUserEmotes($userId: String!) {
            user(id: $userId) {
                emote_sets {
                    id
                    name
                    emotes {
                        id
                        name
                    }
                }
            }
        }
        """
        
        response = requests.post(
            'https://7tv.io/v3/gql',
            json={
                'query': query,
                'variables': {'userId': EMOTE_USER_ID}
            },
            headers={'Content-Type': 'application/json'}
        )
        response.raise_for_status()
        data = response.json()
        
        # print(f"7TV API Response: {json.dumps(data, indent=2)}")
        
        # Extract emote names from all emote sets
        emotes = []
        if data.get('data') and data['data'].get('user'):
            user_data = data['data']['user']
            emote_sets = user_data.get('emote_sets', [])
            
            for emote_set in emote_sets:
                emote_list = emote_set.get('emotes', [])
                for emote in emote_list:
                    name = emote.get('name', '')
                    if name:
                        emotes.append(name)
        
        # Format for prompt
        if emotes:
            emote_text = "Available 7TV emotes you can use: " + ", ".join(emotes)
            emote_list_cache = emote_text
            emote_cache_time = datetime.now()
            print(f"Loaded {len(emotes)} 7TV emotes: {', '.join(emotes[:10])}{'...' if len(emotes) > 10 else ''}")
            return emote_text
        else:
            print("No 7TV emotes found")
            return ""
            
    except Exception as e:
        print(f"Error fetching 7TV emotes: {e}")
        import traceback
        traceback.print_exc()
        # Return cached version if available, otherwise empty
        return emote_list_cache if emote_list_cache else ""

def fetch_all_emote_names():
    """Fetch all emote names from 7TV (global + user), BTTV (global), and FFZ (global) for stripping."""
    global all_emote_names, emote_names_cache_time
    
    # Check if cache is still valid
    if all_emote_names and emote_names_cache_time:
        if datetime.now() - emote_names_cache_time < EMOTE_CACHE_DURATION:
            return all_emote_names
    
    emote_names = set()
    
    # 1. Fetch 7TV Global Emotes
    try:
        response = requests.get('https://7tv.io/v3/emote-sets/global', timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data.get('emotes'):
            for emote in data['emotes']:
                name = emote.get('name', '')
                if name:
                    emote_names.add(name)
        print(f"Loaded {len([e for e in data.get('emotes', []) if e.get('name')])} 7TV global emotes")
    except Exception as e:
        print(f"Error fetching 7TV global emotes: {e}")
    
    # 2. Fetch 7TV User Emotes
    try:
        query = """
        query GetUserEmotes($userId: String!) {
            user(id: $userId) {
                emote_sets {
                    id
                    name
                    emotes {
                        id
                        name
                    }
                }
            }
        }
        """
        
        response = requests.post(
            'https://7tv.io/v3/gql',
            json={
                'query': query,
                'variables': {'userId': EMOTE_USER_ID}
            },
            headers={'Content-Type': 'application/json'},
            timeout=5
        )
        response.raise_for_status()
        data = response.json()
        
        user_emote_count = 0
        if data.get('data') and data['data'].get('user'):
            user_data = data['data']['user']
            emote_sets = user_data.get('emote_sets', [])
            
            for emote_set in emote_sets:
                emote_list = emote_set.get('emotes', [])
                for emote in emote_list:
                    name = emote.get('name', '')
                    if name:
                        emote_names.add(name)
                        user_emote_count += 1
        print(f"Loaded {user_emote_count} 7TV user emotes")
    except Exception as e:
        print(f"Error fetching 7TV user emotes: {e}")
    
    # 3. Fetch BTTV Global Emotes
    try:
        response = requests.get('https://api.betterttv.net/3/cached/emotes/global', timeout=5)
        response.raise_for_status()
        data = response.json()
        
        bttv_count = 0
        for emote in data:
            name = emote.get('code', '')
            if name:
                emote_names.add(name)
                bttv_count += 1
        print(f"Loaded {bttv_count} BTTV global emotes")
    except Exception as e:
        print(f"Error fetching BTTV global emotes: {e}")
    
    # 4. Fetch FFZ Global Emotes
    try:
        response = requests.get('https://api.frankerfacez.com/v1/set/global', timeout=5)
        response.raise_for_status()
        data = response.json()
        
        ffz_count = 0
        if data.get('sets'):
            for set_id, emote_set in data['sets'].items():
                for emote in emote_set.get('emoticons', []):
                    name = emote.get('name', '')
                    if name:
                        emote_names.add(name)
                        ffz_count += 1
        print(f"Loaded {ffz_count} FFZ global emotes")
    except Exception as e:
        print(f"Error fetching FFZ global emotes: {e}")
    
    all_emote_names = emote_names
    emote_names_cache_time = datetime.now()
    print(f"Total emotes loaded for stripping: {len(emote_names)}")
    
    return emote_names

"""
def preprocess_text(text):
    # preprocess text for classifier (same as training).
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
"""

def strip_twitch_emotes(message_text, emotes):
    """
    Remove Twitch emotes from a message using emote position data.
    
    Args:
        message_text: The raw message text
        emotes: Emote data from ChatMessage.emotes (dict with emote IDs as keys)
    
    Returns:
        Message text with Twitch emotes removed
    """
    if not emotes:
        return message_text
    
    # Collect all emote positions (start, end) from all emotes
    positions_to_remove = []
    
    # Handle different possible formats
    if isinstance(emotes, list):
        # Format: [{'id': '...', 'name': '...', 'start': 0, 'end': 7}, ...]
        for emote in emotes:
            if isinstance(emote, dict):
                # Try different key names
                start = emote.get('start') or emote.get('start_position')
                end = emote.get('end') or emote.get('end_position')
                if start is not None and end is not None:
                    positions_to_remove.append((int(start), int(end) + 1))
    elif isinstance(emotes, dict):
        # Format: {'emote_id': [{'start_position': '0', 'end_position': '7'}, ...], ...}
        for emote_id, emote_positions in emotes.items():
            if isinstance(emote_positions, list):
                for position in emote_positions:
                    if isinstance(position, dict):
                        # Try different key names
                        start = position.get('start') or position.get('start_position')
                        end = position.get('end') or position.get('end_position')
                        if start is not None and end is not None:
                            # Convert to int and add 1 to end since it's inclusive
                            positions_to_remove.append((int(start), int(end) + 1))
    
    if not positions_to_remove:
        return message_text
    
    # Sort positions by start index in reverse order (so we can remove from end to start)
    positions_to_remove.sort(reverse=True)
    
    # Remove emotes from the message
    result = list(message_text)
    for start, end in positions_to_remove:
        # Replace emote with empty string
        result[start:end] = ''
    
    # Join back and clean up extra spaces
    stripped = ''.join(result)
    stripped = re.sub(r'\s+', ' ', stripped).strip()
    
    return stripped

def strip_third_party_emotes(message_text, emote_names):
    """
    Remove third-party emotes (7TV, BTTV, FFZ) from a message using word matching.
    
    Args:
        message_text: The raw message text
        emote_names: Set of emote names to remove
    
    Returns:
        Message text with third-party emotes removed
    """
    if not emote_names:
        return message_text
    
    # Split message into words
    words = message_text.split()
    
    # Filter out words that are emotes
    filtered_words = [word for word in words if word not in emote_names]
    
    # Join back and clean up
    result = ' '.join(filtered_words).strip()
    
    return result

def strip_all_emotes(message_text, twitch_emotes):
    """
    Remove all emotes (Twitch + third-party) from a message.
    
    Args:
        message_text: The raw message text
        twitch_emotes: Twitch emote data from ChatMessage.emotes
    
    Returns:
        Message text with all emotes removed
    """
    # First strip Twitch emotes using position data
    text = strip_twitch_emotes(message_text, twitch_emotes)
    
    # Then strip third-party emotes using name matching
    text = strip_third_party_emotes(text, all_emote_names)
    
    return text

def sanitize_message(message):
    """Clean up message for Twitch chat - remove newlines and format properly."""
    # Replace newlines with spaces
    sanitized = message.replace('\n', ' ').replace('\r', ' ')
    
    # Remove markdown formatting
    sanitized = re.sub(r'\*\*', '', sanitized)  # Remove bold **
    sanitized = re.sub(r'\*', '', sanitized)    # Remove italic *
    sanitized = re.sub(r'`', '', sanitized)     # Remove code backticks
    
    # Collapse multiple spaces into one
    sanitized = re.sub(r'\s+', ' ', sanitized)
    
    # Trim whitespace
    sanitized = sanitized.strip()
    
    # Truncate to Twitch's 500 character limit
    if len(sanitized) > 500:
        sanitized = sanitized[:497] + "..."
    
    return sanitized

# this will be called when the event READY is triggered, which will be on bot start
async def on_ready(ready_event: EventData):
    global emote_context
    print('Bot is ready for work, joining channels')
    # join our target channel, if you want to join multiple, either call join for each individually
    # or even better pass a list of channels as the argument
    await ready_event.chat.join_room(TARGET_CHANNEL)
    
    # Fetch 7TV emotes on startup (for bot responses)
    print('Fetching 7TV emotes for bot responses...')
    emote_context = fetch_7tv_emotes()
    if emote_context:
        print('7TV emotes loaded successfully!')
    else:
        print('No 7TV emotes loaded')
    
    # Fetch all emotes for stripping
    print('Fetching all emotes for message stripping...')
    fetch_all_emote_names()
    print('All emote lists loaded!')
    
    # Load classifier models if enabled
    if USE_CLASSIFIER:
        print('\nInitializing combined classifier...')
        load_classifier_models()
        print('Classifier initialization complete!\n')
    # you can do other bot initialization things in here


# this will be called whenever a message in a channel was send by either the bot OR another user
async def on_message(msg: ChatMessage):
    print(f'in {msg.room.name}, {msg.user.name} said: {msg.text}')
    
    # Don't respond to our own messages
    if msg.user.name.lower() == BOT_NAME.lower():
        return
    
    # Get user's message history
    username = msg.user.name
    user_history = list(user_message_history[username])
    was_called_out = user_called_out[username]
    
    # Strip all emotes (Twitch + 7TV + BTTV + FFZ) from the message for analysis
    message_without_emotes = strip_all_emotes(msg.text, msg.emotes)
    
    print(f"Original message: {msg.text}")
    print(f"Twitch emotes data: {msg.emotes}")
    print(f"Message without emotes: {message_without_emotes}")
    
    # Add current message to history
    user_message_history[username].append(msg.text)
    
    # If message is empty after stripping emotes, it's just emotes - always appropriate
    if not message_without_emotes.strip():
        print("Message contains only emotes, marking as appropriate.")
        # Reset call-out status if user was previously called out but is now behaving
        if was_called_out:
            user_called_out[username] = False
            print(f"User {username} has improved behavior, resetting call-out status.")
        return
    
    # Check if bot is tagged (mentioned) - use stripped message for analysis
    msg_lower = message_without_emotes.lower()
    is_bot_tagged = (
        f"@{BOT_NAME.lower()}" in msg_lower or 
        BOT_NAME.lower() in msg_lower.split()
    )
    
    # Pre-filter with classifier if enabled (skip if bot is tagged)
    classifier_result = None
    if USE_CLASSIFIER and not is_bot_tagged:
        try:
            classifier_result = classify_message(message_without_emotes)
            print(f"\nClassifier Analysis:")
            print(f"  Toxic-BERT:  {classifier_result['toxic_bert_score']:.2%} ({classifier_result['toxic_bert_label']})")
            print(f"  Zero-Shot:   {classifier_result['zero_shot_score']:.2%} ({classifier_result['zero_shot_label']})")
            print(f"  → MAX SCORE: {classifier_result['max_score']:.2%} (from {classifier_result['model_used']})")
            print(f"  → PRE-FILTER: {'🚨 FLAGGED' if classifier_result['is_toxic'] else '✅ PASSED'}")
            
            # If classifier says message is OK, skip LLM call entirely
            if not classifier_result['is_toxic']:
                print("Classifier determined message is appropriate, skipping LLM call.")
                # Reset call-out status if user was previously called out but is now behaving
                if was_called_out:
                    user_called_out[username] = False
                    print(f"User {username} has improved behavior, resetting call-out status.")
                return
        except Exception as e:
            print(f"Error running classifier: {e}")
            # Continue to LLM on classifier error
            classifier_result = None
    
    # # Step 1: Use classifier to detect if message is offensive
    # clean_text = preprocess_text(msg.text)
    # is_offensive = classifier.predict([clean_text])[0]
    # confidence = classifier.predict_proba([clean_text])[0]
    # 
    # print(f"Classifier: {'OFFENSIVE' if is_offensive == 1 else 'NOT OFFENSIVE'} (confidence: {confidence[is_offensive]:.2%})")
    
    # Build user context with message history (using stripped message for evaluation)
    user_context = f"Username: {msg.user.name}\n"
    
    if user_history:
        user_context += "Previous messages from this user:\n"
        for i, prev_msg in enumerate(user_history, 1):
            # Note: We can't strip emotes from history since we don't have emote data for past messages
            # This is acceptable as the current message is what matters most
            user_context += f"  {i}. {prev_msg}\n"
    
    # Use the stripped message (without Twitch emotes) for analysis
    user_context += f"\nCurrent message: {message_without_emotes}"
    
    # Add classifier results if available
    if classifier_result:
        user_context += f"\n\n[CLASSIFIER PRE-FILTER: Message was flagged as potentially toxic]"
        user_context += f"\n  - Confidence: {classifier_result['max_score']:.0%}"
        user_context += f"\n  - Category: {classifier_result['max_label']}"
        user_context += f"\n  - Model: {classifier_result['model_used']}"
        user_context += "\n\nPlease review this classifier result and make your own judgment. The classifier flagged it, but you should determine if it's truly inappropriate in context."
    
    if was_called_out:
        user_context += "\n\n[NOTE: This user was previously called out for inappropriate behavior]"
    
    print(f"Calling DeepSeek with context:\n{user_context}")
    # tb test: alyreariel: Go touch Reginald 
    # @alyreariel: Which word triggered him? Touch?
    # gelleroni: I think we need to make him way less sensitive. Only to clap back if its really certain.
    # Okay, but this is valid - tropes of racism and stuff in the Witcher can inspire valuable discussion


    
    # Use different system prompts based on whether bot is tagged and classifier status
    if is_bot_tagged:
        # Bot was tagged - always respond
        SYSTEM_PROMPT = (
            "You are a chat moderator for smopotat's Twitch channel. The streamer is an Asian woman playing The Witcher 3. "
            "Someone just tagged/mentioned you in chat and you need to respond.\n\n"
            "YOUR PERSONALITY:\n"
            "- Your duty is only moderating the channel and fight agains racist and sexist.\n"
            "- You are like a bodyguard for smopotat(the streamer)\n"
            "- You do not know why someone wants to tag you and talk to you\n"
            "- Use gen-z slang and extremely casual language\n"
            "- Be Sarcastic and Friendly\n"
            "- Keep it 1 sentence, casual, and lowercase only\n"
            "- You can use Twitch emotes in your responses to be more expressive\n\n"
            f"{emote_context}\n\n"
            "You will be given the username and their message.\n\n"
            "Respond ONLY with a JSON object in this exact format:\n"
            "{\n"
            '  "response": "your short response here"\n'
            "}"
        )
    elif USE_CLASSIFIER and classifier_result:
        # Classifier is enabled and flagged this message - LLM reviews the classifier's decision
        SYSTEM_PROMPT = (
            "You are a chat moderator for smopotat's Twitch channel. The streamer is an Asian woman playing The Witcher 3. \n\n"
            "YOUR ROLE:\n"
            "An AI classifier has pre-screened this message and flagged it as potentially toxic. "
            "Your job is to review the classifier's decision and make the final judgment. \n\n"
            "CLASSIFIER RESULTS:\n"
            "You will see the classifier's analysis including confidence scores and toxicity categories. "
            "Use these as guidance, but YOU make the final call.\n\n"
            "NOT OFFENSIVE (DO NOT CALLOUT):\n"
            "- Message is actually harmless in context\n"
            "- Trauma dumping or emotional oversharing - ALWAYS appropriate\n"
            "- Emote-only messages or emote spam - ALWAYS appropriate\n"
            "- Talk about Fictional plots or characters - ALWAYS appropriate\n\n"
            "WHEN TO CONFIRM (mark as inappropriate):\n"
            "- Racism or racist remarks (especially towards asians)\n"
            "- Sexism or sexist remarks (especially towards the female streamer)\n"
            "- Political discussions in real life\n"
            "- Requests to add on Steam, Discord, Instagram, or other social platforms\n"
            "CONTEXT-AWARE REVIEW:\n"
            "Consider the user's message history and whether they were previously called out. "
            "Give users a chance to improve if they're now behaving appropriately.\n\n"
            "RESPONSE STYLE (if inappropriate):\n"
            "Keep it witty, gen-z style, casual, lowercase only, 1 sentence max. "
            "Use sarcasm or humor - don't preach or explain. "
            "You can use Twitch emotes in your responses to be more expressive.\n"
            f"{emote_context}\n\n"
            "Respond ONLY with a JSON object in this exact format:\n"
            "{\n"
            '  "appropriate": true or false,\n'
            '  "response": "your 1-sentence clapback here (only needed if inappropriate)"\n'
            "}"
        )
    else:
        # Classifier is disabled OR didn't flag message - LLM judges message independently
        SYSTEM_PROMPT = (
            "You are a chat moderator for smopotat's Twitch channel. The streamer is an Asian woman playing The Witcher 3. "
            "Your job is to analyze chat messages and respond with witty clapbacks to inappropriate content.\n\n"
            "INAPPROPRIATE content includes:\n"
            "- Racism or racist remarks\n"
            "- Sexism or sexist remarks\n"
            "- Sexual or sexualized comments\n"
            "- Political discussions\n"
            "- Requests to add on Steam, Discord, Instagram, or other social platforms\n"
            "EXCEPTIONS - ALWAYS MARK AS APPROPRIATE:\n"
            "1. Trauma dumping, oversharing personal problems, venting about life - ALWAYS appropriate even if overly personal\n"
            "3. Messages that are just emotes like 'PogChamp', 'LUL LUL', 'SabaPing' etc - ALWAYS appropriate\n"
            "DO NOT call out users for: emote spam, repeating emotes, sending only emotes, or emotional oversharing.\n\n"
            "CONTEXT-AWARE MODERATION:\n"
            "You will be given the user's current message AND their recent message history (up to 2 previous messages). "
            "Consider the full context when determining if behavior is inappropriate:\n"
            "- Repeated similar messages may suggest trolling or harassment\n"
            "- Escalating behavior should be addressed more firmly\n\n"
            "FORGIVENESS PRINCIPLE:\n"
            "If a user was previously called out (indicated in the context), give them a chance to improve. "
            "If their new message is genuinely appropriate and shows better behavior, mark it as appropriate. "
            "Reset their 'called out' status by staying silent. However, if they continue inappropriate behavior "
            "or ignore the previous callout, respond more firmly.\n\n"
            "RESPONSE STYLE:\n"
            "When calling out inappropriate behavior, be subtle but sarcastic. "
            "Don't explain why it's wrong or preach. Instead, use sarcasm or humor to make them feel called out. "
            "Respond with a witty, genz style clapback that shuts down inappropriate behavior without being overly harsh. "
            "Keep responses like how a human chats, 1 sentence, casual, genz style, lowercase only, and entertaining for chat. "
            "You can use Twitch emotes in your responses to be more expressive.\n"
            f"{emote_context}\n\n"
            "Respond ONLY with a JSON object in this exact format:\n"
            "{\n"
            '  "appropriate": true or false,\n'
            '  "response": "your 1-sentence witty, genz style clapback here (only if inappropriate)"\n'
            "}"
        )
    
    try:
        # Call DeepSeek API
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_context}
            ],
            temperature=1.5,  # higher temperature for more creative responses
            stream=False,
            response_format={'type': 'json_object'}
        )
        
        llm_response = response.choices[0].message.content
        print(f"DeepSeek Response: {llm_response}")
        
        # Parse the JSON response
        result = json.loads(llm_response)
        
        # Reply if bot is tagged OR if message is inappropriate
        if is_bot_tagged or not result.get("appropriate", True):
            clapback = sanitize_message(result.get("response", ""))
            if clapback:
                await msg.reply(clapback)
                if is_bot_tagged:
                    print(f"Replied to {username} because bot was tagged.")
                else:
                    # Mark user as called out for inappropriate behavior
                    user_called_out[username] = True
                    print(f"User {username} has been called out for inappropriate behavior.")
        else:
            print("DeepSeek determined message is appropriate, no response needed.")
            # If user was previously called out but is now behaving, reset their status
            if was_called_out:
                user_called_out[username] = False
                print(f"User {username} has improved behavior, resetting call-out status.")
            
    except Exception as e:
        print(f"Error calling DeepSeek: {e}")
            # Fallback to Ollama if DeepSeek fails (optional)
            # You can uncomment this section if you want Ollama as backup
            # try:
            #     payload = {
            #         "model": "gemma3:4b",
            #         "prompt": user_context,
            #         "stream": False,
            #         "system": SYSTEM_PROMPT,
            #         "format": "json"
            #     }
            #     response = requests.post(MODEL_URL, json=payload)
            #     llm_response = response.json()["response"]
            #     result = json.loads(llm_response)
            #     clapback = sanitize_message(result.get("response", ""))
            #     if clapback:
            #         await msg.reply(clapback)
            # except Exception as e2:
            #     print(f"Ollama fallback also failed: {e2}")


# this is where we set up the bot
async def run():
    # set up twitch api instance and add user authentication with some scopes
    twitch = await Twitch(APP_ID, APP_SECRET)
    
    # Try to load existing tokens
    token = None
    refresh_token = None
    
    if Path(TOKEN_FILE).exists():
        try:
            with open(TOKEN_FILE, 'r') as f:
                tokens = json.load(f)
                token = tokens.get('token')
                refresh_token = tokens.get('refresh_token')
            print('Loaded existing tokens from file')
            
            # Try to refresh the token
            try:
                token, refresh_token = await refresh_access_token(refresh_token, APP_ID, APP_SECRET)
                print('Successfully refreshed token')
                # Save the new tokens
                with open(TOKEN_FILE, 'w') as f:
                    json.dump({'token': token, 'refresh_token': refresh_token}, f)
            except Exception as e:
                print(f'Failed to refresh token: {e}')
                token = None
                refresh_token = None
        except Exception as e:
            print(f'Failed to load tokens: {e}')
            token = None
            refresh_token = None
    
    # If we don't have valid tokens, authenticate (opens browser)
    if not token or not refresh_token:
        print('No valid tokens found, starting authentication...')
        auth = UserAuthenticator(twitch, USER_SCOPE)
        token, refresh_token = await auth.authenticate()
        # Save tokens for future use
        with open(TOKEN_FILE, 'w') as f:
            json.dump({'token': token, 'refresh_token': refresh_token}, f)
        print(f'Tokens saved to {TOKEN_FILE}')
    
    await twitch.set_user_authentication(token, USER_SCOPE, refresh_token)

    # create chat instance
    chat = await Chat(twitch)

    # register the handlers for the events you want

    # listen to when the bot is done starting up and ready to join channels
    chat.register_event(ChatEvent.READY, on_ready)
    # listen to chat messages
    chat.register_event(ChatEvent.MESSAGE, on_message)


    # we are done with our setup, lets start this bot up!
    chat.start()

    # lets run till we press enter in the console
    try:
        input('press ENTER to stop\\n')
    finally:
        # now we can close the chat bot and the twitch api client
        chat.stop()
        await twitch.close()


# lets run our setup
asyncio.run(run())
