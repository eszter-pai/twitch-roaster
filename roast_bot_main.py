import re
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from twitchAPI.twitch import Twitch
from twitchAPI.oauth import UserAuthenticator, refresh_access_token
from twitchAPI.type import AuthScope, ChatEvent
from twitchAPI.chat import Chat, EventData, ChatMessage, ChatSub, ChatCommand
import asyncio
from collections import deque, defaultdict
from openai import OpenAI
from datetime import datetime, timedelta
from prompts import (
    get_bot_tagged_prompt,
    get_classifier_review_prompt,
    get_independent_judge_prompt,
    build_user_context
)
from emote_handler import (
    fetch_7tv_emotes,
    fetch_all_emote_names,
    strip_all_emotes,
    get_all_emote_names
)
from classifier import (
    load_classifier_models,
    classify_message,
    is_classifier_loaded
)

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
USE_CLASSIFIER = os.getenv('USE_CLASSIFIER', 'true').lower() == 'true'
CLASSIFIER_THRESHOLD = float(os.getenv('CLASSIFIER_THRESHOLD', '0.88'))

# Initialize DeepSeek client
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# Message history storage
general_chat_history = deque(maxlen=10)  # Last 10 messages from all users
user_message_history = defaultdict(lambda: deque(maxlen=3))  # Last 3 messages per user
user_called_out = defaultdict(bool)  # Track if user was recently called out

# 7TV Emote context for LLM
emote_context = ""  # Global variable to store emote context

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
    emote_context = fetch_7tv_emotes(EMOTE_USER_ID)
    if emote_context:
        print('7TV emotes loaded successfully!')
    else:
        print('No 7TV emotes loaded')
    
    # Fetch all emotes for stripping
    print('Fetching all emotes for message stripping...')
    fetch_all_emote_names(EMOTE_USER_ID)
    print('All emote lists loaded!')
    
    # Load classifier models if enabled
    if USE_CLASSIFIER:
        print('\nInitializing combined classifier...')
        try:
            load_classifier_models(CLASSIFIER_THRESHOLD)
            print('Classifier initialization complete!\n')
        except Exception as e:
            print(f'Failed to load classifier: {e}')
            print('Continuing without classifier...\n')
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
    message_without_emotes = strip_all_emotes(msg.text, msg.emotes, get_all_emote_names())
    
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
            classifier_result = classify_message(message_without_emotes, CLASSIFIER_THRESHOLD)
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
    user_context = build_user_context(
        username=msg.user.name,
        user_history=user_history,
        current_message=message_without_emotes,
        classifier_result=classifier_result,
        was_called_out=was_called_out
    )
    
    print(f"Calling DeepSeek with context:\n{user_context}")
    # tb test: alyreariel: Go touch Reginald 
    # @alyreariel: Which word triggered him? Touch?
    # gelleroni: I think we need to make him way less sensitive. Only to clap back if its really certain.
    # Okay, but this is valid - tropes of racism and stuff in the Witcher can inspire valuable discussion


    
    # Use different system prompts based on whether bot is tagged and classifier status
    if is_bot_tagged:
        # Bot was tagged - always respond
        SYSTEM_PROMPT = get_bot_tagged_prompt(emote_context)
    elif USE_CLASSIFIER and classifier_result:
        # Classifier is enabled and flagged this message - LLM reviews the classifier's decision
        SYSTEM_PROMPT = get_classifier_review_prompt(emote_context)
    else:
        # Classifier is disabled OR didn't flag message - LLM judges message independently
        SYSTEM_PROMPT = get_independent_judge_prompt(emote_context)
    
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
