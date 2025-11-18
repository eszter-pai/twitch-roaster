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
from datetime import datetime, timedelta

load_dotenv()

BOT_NAME = os.getenv('TWITCH_BOT_USERNAME')
MODEL_URL = os.getenv('MODEL_API_URL')  # Ollama URL (kept for fallback)
APP_ID  = os.getenv('CLIENT_ID')
APP_SECRET  = os.getenv('CLIENT_SECRET')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
USER_SCOPE = [AuthScope.CHAT_READ, AuthScope.CHAT_EDIT]
TARGET_CHANNEL = os.getenv('TWITCH_CHANNEL')
TOKEN_FILE = 'twitch_tokens.json'
CLASSIFIER_MODEL = 'offensive_classifier.joblib' # this is the trained logreg model


# Load the trained classifier
# print('Loading offensive message classifier...')
# classifier = joblib.load(CLASSIFIER_MODEL)
# print('Classifier loaded successfully!')

# Initialize DeepSeek client
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# Message history storage
general_chat_history = deque(maxlen=10)  # Last 10 messages from all users
user_message_history = defaultdict(lambda: deque(maxlen=3))  # Last 3 messages per user
user_called_out = defaultdict(bool)  # Track if user was recently called out

# 7TV Emote cache
emote_list_cache = None
emote_cache_time = None
EMOTE_CACHE_DURATION = timedelta(hours=1)  # Refresh every hour

def fetch_7tv_emotes():
    """Fetch emotes from 7TV API and format them for the LLM."""
    global emote_list_cache, emote_cache_time
    
    # Check if cache is still valid
    if emote_list_cache and emote_cache_time:
        if datetime.now() - emote_cache_time < EMOTE_CACHE_DURATION:
            return emote_list_cache
    
    try:
        # Fetch from 7TV API
        response = requests.get(f'https://7tv.io/v3/users/{SEVENTV_USER_ID}')
        response.raise_for_status()
        data = response.json()
        
        # Extract emote names and descriptions
        emotes = []
        emote_set = data.get('emote_set', {})
        for emote in emote_set.get('emotes', []):
            name = emote.get('name', '')
            # Some emotes have descriptions/tags that might be useful
            if name:
                emotes.append(name)
        
        # Format for prompt
        if emotes:
            emote_text = "Available 7TV emotes you can use: " + ", ".join(emotes)
            emote_list_cache = emote_text
            emote_cache_time = datetime.now()
            print(f"Loaded {len(emotes)} 7TV emotes")
            return emote_text
        else:
            return ""
            
    except Exception as e:
        print(f"Error fetching 7TV emotes: {e}")
        # Return cached version if available, otherwise empty
        return emote_list_cache if emote_list_cache else ""

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
    
    # Get user's message history
    username = msg.user.name
    user_history = list(user_message_history[username])
    was_called_out = user_called_out[username]
    
    # Add current message to history
    user_message_history[username].append(msg.text)
    
    # Check if bot is tagged (mentioned)
    is_bot_tagged = f"@{BOT_NAME}" in msg.text.lower() or BOT_NAME in msg.text.lower().split()
    
    # # Step 1: Use classifier to detect if message is offensive
    # clean_text = preprocess_text(msg.text)
    # is_offensive = classifier.predict([clean_text])[0]
    # confidence = classifier.predict_proba([clean_text])[0]
    # 
    # print(f"Classifier: {'OFFENSIVE' if is_offensive == 1 else 'NOT OFFENSIVE'} (confidence: {confidence[is_offensive]:.2%})")
    
    # Build user context with message history
    user_context = f"Username: {msg.user.name}\n"
    
    if user_history:
        user_context += "Previous messages from this user:\n"
        for i, prev_msg in enumerate(user_history, 1):
            user_context += f"  {i}. {prev_msg}\n"
    
    user_context += f"\nCurrent message: {msg.text}"
    
    if was_called_out:
        user_context += "\n\n[NOTE: This user was previously called out for inappropriate behavior]"
    
    print(f"Calling DeepSeek with context:\n{user_context}")
    
    # Fetch available emotes
    # emote_context = fetch_7tv_emotes()
    
    # Use different system prompts based on whether bot is tagged
    if is_bot_tagged:
        SYSTEM_PROMPT = (
            "You are a chat moderator for smopotat's Twitch channel. The streamer is an Asian woman playing The Witcher 3. "
            "Someone just tagged/mentioned you in chat and you need to respond.\n\n"
            "YOUR PERSONALITY:\n"
            "- You don't want to encourage people to tag you\n"
            "- Your duty is only moderating the channel and fight agains racist and sexist, so you do not know why someone wants to tag you and talk to you.\n"
            "- Use gen-z slang and extremely casual language\n"
            "- Be sarcastic but friendly\n"
            "- Keep it 1 sentence, casual, and lowercase only\n"
        #    "- You can use Twitch emotes in your responses to be more expressive\n\n"
        #    f"{emote_context}\n\n"
            "You will be given the username and their message. Respond as if you're annoyed they interrupted your gameplay.\n\n"
            "Respond ONLY with a JSON object in this exact format:\n"
            "{\n"
            '  "response": "your super short annoyed response here"\n'
            "}"
        )
    else:
        SYSTEM_PROMPT = (
            "You are a chat moderator for smopotat's Twitch channel. The streamer is an Asian woman playing The Witcher 3. "
            "Your job is to respond with witty, clever clapbacks to inappropriate messages and protect her from harmful statements.\n\n"
            "INAPPROPRIATE content includes:\n"
            "- Racism or racist remarks\n"
            "- Sexism or sexist remarks\n"
            "- Sexual or sexualized comments\n"
            "- Political discussions\n"
        #    "- Harassment or targeted attacks\n"
            "- Requests to add on Steam, Discord, Instagram, or other social platforms\n\n"
            "IMPORTANT: Do NOT consider trauma dumping, oversharing personal problems, or emote spamming as inappropriate. "
            "These messages should be marked as appropriate even if they're overly personal or emotional.\n\n"
            "CONTEXT-AWARE MODERATION:\n"
            "You will be given the user's current message AND their recent message history (up to 2 previous messages). "
            "Consider the full context when determining if behavior is inappropriate:\n"
            "- A pattern of borderline comments may indicate inappropriate intent\n"
            "- Repeated similar messages may suggest trolling or harassment\n"
            "- Escalating behavior should be addressed more firmly\n\n"
            "FORGIVENESS PRINCIPLE:\n"
            "If a user was previously called out (indicated in the context), give them a chance to improve. "
            "If their new message is genuinely appropriate and shows better behavior, mark it as appropriate. "
            "Reset their 'called out' status by staying silent. However, if they continue inappropriate behavior "
            "or ignore the previous callout, respond more firmly.\n\n"
            "Respond with a witty, genz style clapback that shuts down inappropriate behavior without being overly harsh. "
            "Keep responses like how a human chats, 1 sentence, casual, genz style, lowercase only, and entertaining for chat. "
        #    "You can use Twitch emotes in your responses to be more expressive.\n\n"
        #    f"{emote_context}\n\n"
            "Respond ONLY with a JSON object in this exact format:\n"
            "{\n"
            '  "appropriate": false,\n'
            '  "response": "your 1-sentence witty, genz style clapback here"\n'
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
