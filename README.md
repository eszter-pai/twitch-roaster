# Twitch Roaster Bot

A Twitch chat moderation bot that uses AI to detect and respond to inappropriate messages with witty clapbacks.

## Current Features

- Real-time Twitch chat monitoring
- AI-powered message evaluation (Used so far: gemma3:4b locally and Deepseek using API key)
- Automatic witty clapback responses to inappropriate content
- Detects racism, sexism, sexual comments, harassment, and more

## Known Issues

### 1. ML Classifier Limitations
- In order to save resources, I initially trained a simple ML classifier on Twitter hate speech data to filter messages before passing them to the LLM.
- The trained ML classifier (using Twitter hate speech data) cannot detect subtle sexism/racism like "do you eat dogs" or "is it pink". 

**Current workaround:** Using DeepSeek API for all message evaluation instead.
**Things I can try:** Zero shot classification? Trasformer Models (HateBERT)?

### 2. Ollama Performance Issues
Running Ollama with gemma3:4b locally uses too many resources to run while streaming.

**Current solution:** Switched to DeepSeek API which runs remotely.

### 3. RAG Implementation
Want to use RAG (Retrieval-Augmented Generation) in this project but still researching how to implement it effectively.

### 4. Chat History Context
Need smarter chat history handling:
- Consider context for detecting patterns and escalation
- Ignore history when user improves behavior after being called out
- Balance between context-awareness and forgiveness

**Ideas to try:**
- Add "forgiveness" logic to system prompt

## To-Do

- [ ] Figure out RAG implementation
- [ ] Add intelligent chat history tracking
- [ ] Improve system prompt for nuanced decisions
- [ ] teach bot use twitch emotes in responses 
- [ ] abstract API calling logics to a separate module (so i can switch between different LLM providers just using.env variables)
- [ ] zero shot classification?
- [ ] Bot should reply when it is tagged, no matter if it is appropriate or not
- [ ] Host bot on a server?
- [ ] Web UI

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Configure `.env`:
   ```
   TWITCH_BOT_USERNAME=your_bot_username
   CLIENT_ID=your_twitch_client_id
   CLIENT_SECRET=your_twitch_client_secret
   DEEPSEEK_API_KEY=your_deepseek_api_key
   TWITCH_CHANNEL=target_channel_name
   ```
3. Run: `python simple_roast_bot.py`