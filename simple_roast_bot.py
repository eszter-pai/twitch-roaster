import socket
import re
import os
import json
from pathlib import Path
from dotenv import load_dotenv
import requests
from twitchAPI.twitch import Twitch
from twitchAPI.oauth import UserAuthenticator, refresh_access_token
from twitchAPI.type import AuthScope, ChatEvent
from twitchAPI.chat import Chat, EventData, ChatMessage, ChatSub, ChatCommand
import asyncio
import random
from collections import deque, defaultdict
import joblib
from openai import OpenAI

load_dotenv()

BOT_NAME = os.getenv('TWITCH_BOT_USERNAME')
MODEL_URL = os.getenv('MODEL_API_URL')  # Ollama URL (kept for fallback)
APP_ID  = os.getenv('CLIENT_ID')
APP_SECRET  = os.getenv('CLIENT_SECRET')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
USER_SCOPE = [AuthScope.CHAT_READ, AuthScope.CHAT_EDIT]
TARGET_CHANNEL = os.getenv('TWITCH_CHANNEL')
TOKEN_FILE = 'twitch_tokens.json'
CLASSIFIER_MODEL = 'offensive_classifier.joblib'

# Load the trained classifier
# print('Loading offensive message classifier...')
# classifier = joblib.load(CLASSIFIER_MODEL)
# print('Classifier loaded successfully!')

# Initialize DeepSeek client
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# Message history storage
general_chat_history = deque(maxlen=10)  # Last 10 messages from all users

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
    print('Bot is ready for work, joining channels')
    # join our target channel, if you want to join multiple, either call join for each individually
    # or even better pass a list of channels as the argument
    await ready_event.chat.join_room(TARGET_CHANNEL)
    # you can do other bot initialization things in here


# this will be called whenever a message in a channel was send by either the bot OR another user
async def on_message(msg: ChatMessage):
    print(f'in {msg.room.name}, {msg.user.name} said: {msg.text}')
    
    # Don't respond to our own messages
    if msg.user.name.lower() == BOT_NAME.lower():
        return
    
    # # Step 1: Use classifier to detect if message is offensive
    # clean_text = preprocess_text(msg.text)
    # is_offensive = classifier.predict([clean_text])[0]
    # confidence = classifier.predict_proba([clean_text])[0]
    # 
    # print(f"Classifier: {'OFFENSIVE' if is_offensive == 1 else 'NOT OFFENSIVE'} (confidence: {confidence[is_offensive]:.2%})")
    
    # Call DeepSeek to determine if message is offensive and get clapback
    user_context = f"Username: {msg.user.name}\nMessage: {msg.text}"
    print(f"Calling DeepSeek with context:\n{user_context}")
    
    SYSTEM_PROMPT = (
        "You are a chat moderator for smopotat's Twitch channel. The streamer is an Asian woman playing The Witcher 3. "
        "Your job is to respond with witty, clever clapbacks to inappropriate messages.\n\n"
        "INAPPROPRIATE content includes:\n"
        "- Racism or racist remarks\n"
        "- Sexism or sexist remarks\n"
        "- Sexual or sexualized comments\n"
        "- Political discussions\n"
        "- Harassment or targeted attacks\n"
        "- Requests to add on Steam, Discord, Instagram, or other social platforms\n"
        "- Trauma dumping or oversharing personal problems\n\n"
        "You will be given the username and their message. "
        "Determine if the message is inappropriate based on the criteria above.\n\n"
        "Respond with a clever, witty clapback that shuts down the inappropriate behavior without being overly harsh. "
        "Keep responses short, punchy, and entertaining for chat.\n\n"
        "Respond ONLY with a JSON object in this exact format:\n"
        "{\n"
        '  "appropriate": false,\n'
        '  "response": "your witty clapback here"\n'
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
            stream=False
        )
        
        llm_response = response.choices[0].message.content
        print(f"DeepSeek Response: {llm_response}")
        
        # Parse the JSON response
        result = json.loads(llm_response)
        
        # Only send clapback if DeepSeek determines message is inappropriate
        if not result.get("appropriate", True):
            clapback = sanitize_message(result.get("response", ""))
            if clapback:
                await msg.reply(clapback)
        else:
            print("DeepSeek determined message is appropriate, no response needed.")
            
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
