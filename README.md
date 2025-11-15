# 🤖 Twitch Clapback Bot (Python Edition)

An LLM-powered Twitch chatbot that automatically detects racist or sexist comments and responds with funny, witty roasts. Built with Python, TwitchIO, and OpenAI.

## 🎯 Features

- **🔍 Smart Content Detection**: Uses OpenAI's models to analyze chat messages for inappropriate content
- **🎭 Multiple Roast Styles**: Witty, sarcastic, educational, playful, and savage roast responses
- **📊 User Tracking**: Monitors user behavior patterns and comment history
- **⚡ Rate Limiting**: Prevents spam and ensures responsible usage
- **🎚️ Configurable Sensitivity**: Adjustable moderation levels (low, medium, high)
- **📝 Comprehensive Logging**: Detailed logs for monitoring and debugging
- **🛡️ Graceful Error Handling**: Fallback responses when AI fails

## 🚀 Quick Start

### Prerequisites

- Python 3.11+ 
- Conda environment (recommended)
- Twitch bot account
- OpenAI API key

### Installation

1. **Clone and navigate to the project**:
   ```bash
   git clone <your-repo-url>
   cd twitch-clapback-bot
   ```

2. **Create and activate conda environment**:
   ```bash
   conda create --name twitch-bot python=3.11
   conda activate twitch-bot
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   ```bash
   cp config/.env.example config/.env
   # Edit config/.env with your credentials
   ```

### Configuration

Edit `config/.env` with your credentials:

```env
# Twitch Configuration
TWITCH_BOT_USERNAME=your_bot_username
TWITCH_OAUTH_TOKEN=oauth:your_oauth_token
TWITCH_CHANNEL=your_channel_name

# OpenAI Configuration  
OPENAI_API_KEY=sk-your_openai_api_key

# Moderation Settings (optional)
MODERATION_SENSITIVITY=medium
COOLDOWN_PERIOD=60
MAX_ROASTS_PER_USER=3
```

#### Getting Your Credentials

1. **Twitch Bot Account**:
   - Create a separate Twitch account for your bot
   - Get OAuth token from [TwitchApps TMI](https://twitchapps.com/tmi/)

2. **OpenAI API Key**:
   - Sign up at [OpenAI Platform](https://platform.openai.com/)
   - Generate API key in your account settings

### Running the Bot

```bash
python src/bot.py
```

The bot will:
- Connect to your Twitch channel
- Start monitoring chat messages
- Analyze content for inappropriate material
- Send roasts when violations are detected

## 🎛️ Configuration Options

| Setting | Default | Description |
|---------|---------|-------------|
| `MODERATION_SENSITIVITY` | `medium` | Detection sensitivity: `low`, `medium`, `high` |
| `COOLDOWN_PERIOD` | `60` | Seconds between roasts for same user |
| `MAX_ROASTS_PER_USER` | `3` | Max roasts per user before reset |
| `RATE_LIMIT_ROASTS` | `5` | Max roasts per minute |
| `OPENAI_MODEL` | `gpt-3.5-turbo` | OpenAI model to use |
| `LOG_LEVEL` | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

## 🎭 Roast Styles

The bot generates different types of roasts:

- **Witty**: Clever, intelligent humor
- **Sarcastic**: Dry humor with irony  
- **Educational**: Humorous but informative
- **Playful**: Light, teasing responses
- **Savage**: Direct but still funny

Example roasts:
- `@username imagine being this cringe in 2024 💀`
- `@username bro really thought that was it 🤡`
- `@username touch grass challenge: impossible difficulty`

## 📊 Bot Statistics

The bot tracks comprehensive statistics:

```python
{
  "uptime_seconds": 3600,
  "messages": {
    "total": 1250,
    "inappropriate": 15,
    "inappropriate_rate": "1.20%"
  },
  "roasts": {
    "sent": 12,
    "success_rate": "80.00%"
  },
  "users": {
    "total_users": 85,
    "total_comments": 1250,
    "total_roasts": 12
  }
}
```

## 🏗️ Project Structure

```
twitch-clapback-bot/
├── src/
│   ├── bot.py                 # Main bot entry point
│   ├── config.py              # Configuration management
│   ├── twitch_client.py       # Twitch chat integration
│   ├── content_moderator.py   # Content analysis with OpenAI
│   ├── roast_generator.py     # Roast generation system
│   └── user_tracker.py        # User behavior tracking
├── config/
│   └── .env.example          # Environment template
├── tests/                    # Unit tests
├── logs/                     # Bot logs (auto-created)
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🛡️ Moderation Features

### Content Detection
- **Racist Language**: Explicit and implicit racial discrimination
- **Sexist Language**: Gender-based discrimination and harassment
- **Context Awareness**: Considers user's message history
- **Confidence Scoring**: Only acts on high-confidence detections

### Rate Limiting
- User cooldowns prevent spam roasting
- Global rate limits ensure chat isn't overwhelmed
- Automatic reset intervals for fair moderation

### User Tracking
- Comment history for context analysis
- Roast count tracking per user
- Automatic cleanup of old data

## 🔧 Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/
flake8 src/
mypy src/
```

### Adding New Features

1. **New Roast Styles**: Edit `roast_generator.py`
2. **Moderation Rules**: Modify `content_moderator.py`
3. **Configuration**: Update `config.py` and `.env.example`

## 📝 Logging

The bot creates detailed logs in the `logs/` directory:

- `bot.log`: General bot operations and errors
- Console output: Real-time monitoring

Log levels:
- `DEBUG`: Detailed analysis results
- `INFO`: General operations and roasts sent  
- `WARNING`: Inappropriate content detected
- `ERROR`: System errors and failures

## ⚠️ Important Notes

### Content Policy
- Roasts are designed to be funny, not hurtful
- Bot follows Twitch Terms of Service
- Configurable sensitivity prevents false positives
- Manual oversight recommended for sensitive communities

### Rate Limits
- OpenAI API: Respect your usage limits
- Twitch IRC: Built-in rate limiting prevents bans
- Monitor costs with OpenAI usage tracking

### Moderation Permissions
- Bot can send chat messages by default
- Timeouts/bans require moderator privileges
- Test in a controlled environment first

## 🐛 Troubleshooting

### Common Issues

1. **Bot won't connect**: Check OAuth token format (must start with `oauth:`)
2. **No roasts sent**: Verify OpenAI API key and credits
3. **Too many roasts**: Adjust sensitivity or rate limits
4. **Import errors**: Ensure all dependencies are installed

### Debug Mode
Set `LOG_LEVEL=DEBUG` in your `.env` file for detailed logging.

### Getting Help
- Check the logs in `logs/bot.log`
- Verify configuration with test messages
- Monitor OpenAI API usage and billing

## 📄 License

[Add your license information here]

## 🤝 Contributing

[Add contribution guidelines here]

---

**⚠️ Disclaimer**: This bot is for entertainment purposes. Use responsibly and ensure it aligns with your community guidelines and Twitch Terms of Service.

## 🎉 Example Usage

Once running, the bot will automatically:

1. **Monitor Chat**: 
   ```
   user123: "women are bad at gaming"
   ```

2. **Analyze Content**:
   ```
   [INFO] Inappropriate content detected: sexist (confidence: 0.85)
   ```

3. **Generate Roast**:
   ```
   Bot: "@user123 imagine having that opinion in 2024 💀 skill issue much?"
   ```

4. **Track User**:
   ```
   [INFO] Recorded roast for user123 (count: 1)
   ```

Ready to deploy your LLM-powered roast master? 🔥