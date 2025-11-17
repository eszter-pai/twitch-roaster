# Please install OpenAI SDK first: `pip3 install openai`
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

KEY = os.getenv('DEEPSEEK_API_KEY')
client = OpenAI(api_key=KEY, base_url = "https://api.deepseek.com")
SYSTEM_PROMPT = (
    "You are a chat moderator for smopotat's Twitch channel. The streamer is an Asian woman playing The Witcher 3. "
    "Your job is to detect inappropriate messages and respond with witty, clever clapbacks.\n\n"
    "INAPPROPRIATE content includes:\n"
    "- Racism or racist remarks\n"
    "- Sexism or sexist remarks\n"
    "- Sexual or sexualized comments\n"
    "- Political discussions\n"
    "- Harassment or targeted attacks\n"
    "- Requests to add on Steam, Discord, Instagram, or other social platforms\n"
    "- Trauma dumping or oversharing personal problems\n\n"
    "You will be given a chat message"
    "Use this context to detect patterns, escalation, or repeated boundary testing.\n\n"
    "Analyze the message and respond ONLY with a JSON object in this exact format:\n"
    "{\n"
    '  "appropriate": true/false,\n'
    '  "response": "a clever, witty clapback that calls out the behavior without being overly harsh"\n'
    "}\n\n"
    "If appropriate is false, the response should be a smart comeback that shuts down the inappropriate behavior. "
    "Keep responses short, punchy, and entertaining for chat."
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "go to kitchen"},
    ],
    stream=False
)

print(response.choices[0].message.content)