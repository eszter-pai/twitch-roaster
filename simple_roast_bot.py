import socket
import re
import os
from dotenv import load_dotenv
import requests
from twitchAPI.twitch import Twitch
from twitchAPI.oauth import UserAuthenticator
from twitchAPI.type import AuthScope, ChatEvent
from twitchAPI.chat import Chat, EventData, ChatMessage, ChatSub, ChatCommand
import asyncio

load_dotenv()

BOT_NAME = os.getenv('TWITCH_BOT_USERNAME')
MODEL_URL = os.getenv('MODEL_API_URL')
APP_ID  = os.getenv('CLIENT_ID')
APP_SECRET  = os.getenv('CLIENT_SECRET')
USER_SCOPE = [AuthScope.CHAT_READ, AuthScope.CHAT_EDIT]
TARGET_CHANNEL = os.getenv('TWITCH_CHANNEL')

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


# this is where we set up the bot
async def run():
    # set up twitch api instance and add user authentication with some scopes
    twitch = await Twitch(APP_ID, APP_SECRET)
    auth = UserAuthenticator(twitch, USER_SCOPE)
    token, refresh_token = await auth.authenticate()
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


    
def sanitize_message(self, message):
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

   


def handle_message(self, msg_data):
    """Handle incoming chat message and check for inappropriate content."""
    if not msg_data:
        return
    
    self.message_count += 1
    print(msg_data)
    username = msg_data['username']
    message = msg_data['message']
    
    print(f"\n[MSG #{self.message_count}] {username}: {message}")
    
    # Don't respond to our own messages
    if username.lower() == self.nickname.lower():
        return
    
    SYSTEM_PROMPT = (
    "You are an offensive-language detector. "
    "Given a user message, determine whether it is offensive. "
    "Always respond ONLY with a JSON dictionary like this:\n\n"
    "{\n"
    "  \"offensive\": true/false,\n"
    "  \"response\": \"a witty clapback\"\n"
    "}\n\n"
    "Do NOT include anything else."
    )

    payload = {
        "model": "gemma3:4b",
        "prompt": message,
        "stream": False,
        "system": SYSTEM_PROMPT
    }

    response = requests.post(MODEL_URL, json=payload)
    print(response.json()["response"])


    # self.send_chat_message(response.json()["response"])
