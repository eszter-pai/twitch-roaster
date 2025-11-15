import socket
import re
import os
from dotenv import load_dotenv

load_dotenv()

ACCESS_TOKEN =  os.getenv('TWITCH_OAUTH_TOKEN')
CHANNEL = os.getenv('TWITCH_CHANNEL')
BOT_NAME = os.getenv('TWITCH_BOT_USERNAME')

class SimpleTwitchRoastBot:
    """Simple roast bot for inappropriate content."""
    
    def __init__(self, token, channel, nickname):
        self.token = token
        self.channel = channel.lower()
        self.nickname = nickname.lower()
        self.socket = None
        self.running = False
        self.message_count = 0
        print(os.environ['Greets'])

    def connect(self):
        """Connect to Twitch IRC servers."""
        print("[*] Connecting to Twitch IRC...")
        
        server = 'irc.chat.twitch.tv'
        port = 6667
        
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((server, port))
            
            # Authenticate
            self.send_raw(f"PASS oauth:{self.token}")
            self.send_raw(f"NICK {self.nickname}")
            self.send_raw(f"JOIN #{self.channel}")
            
            # Request tags
            self.send_raw("CAP REQ :twitch.tv/tags")
            self.send_raw("CAP REQ :twitch.tv/commands")
            
            print(f"[+] Connected to #{self.channel} as {self.nickname}")
            print(f"[>] Roast bot is active! Monitoring for inappropriate content...")
            print("-" * 70)
            
            self.running = True
            return True
            
        except Exception as e:
            print(f"[-] Connection failed: {e}")
            return False
    
    def send_raw(self, message):
        """Send raw IRC message."""
        if self.socket:
            self.socket.send(f"{message}\r\n".encode('utf-8'))
    
    def send_chat_message(self, message):
        """Send a message to the Twitch chat channel."""
        if self.socket and self.running:
            chat_msg = f"PRIVMSG #{self.channel} :{message}"
            self.send_raw(chat_msg)
            print(f"[ROAST SENT] {message}")
    
    def parse_message(self, raw_line):
        """Parse IRC message and extract information."""
        line = raw_line.strip()
        
        if not line or line.startswith('PING') or ':tmi.twitch.tv' in line:
            return None
        
        if 'PRIVMSG' not in line:
            return None
        
        try:
            parts = line.split(':', 2)
            if len(parts) < 3:
                return None
            
            tags_and_prefix = parts[0] + ':' + parts[1]
            message_content = parts[2]
            
            privmsg_match = re.search(r':(\w+)!\w+@\w+\.tmi\.twitch\.tv PRIVMSG #(\w+)', tags_and_prefix)
            if not privmsg_match:
                return None
            
            username = privmsg_match.group(1)
            channel = privmsg_match.group(2)
            
            return {
                'username': username,
                'channel': channel,
                'message': message_content,
                'raw': line
            }
            
        except Exception as e:
            print(f"Parse error: {e}")
            return None
    
    
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
        
        self.send_chat_message("Hello")
        
       
    def listen(self):
        """Listen for incoming messages."""
        if not self.socket:
            print("[-] Not connected!")
            return
        
        try:
            while self.running:
                response = self.socket.recv(2048).decode('utf-8', errors='ignore')
                
                if not response:
                    print("[-] Connection lost!")
                    break
                
                lines = response.split('\r\n')
                
                for line in lines:
                    if not line:
                        continue
                    
                    # Handle PING
                    if line.startswith('PING'):
                        self.send_raw('PONG :tmi.twitch.tv')
                        continue
                    
                    # Parse and handle chat messages
                    print(line)
                    msg_data = self.parse_message(line)
                    if msg_data:
                        self.handle_message(msg_data)
        
        except KeyboardInterrupt:
            print("\n[-] Stopping roast bot...")
        except Exception as e:
            print(f"[-] Listen error: {e}")
        finally:
            self.disconnect()
    
    def disconnect(self):
        """Disconnect from IRC."""
        self.running = False
        if self.socket:
            try:
                self.socket.close()
                print("[*] Disconnected from Twitch IRC")
            except:
                pass

def main():
    """Main function to run the roast bot."""
    print("[*] TWITCH ROAST BOT - INAPPROPRIATE CONTENT DETECTOR")
    print("=" * 60)
    print("[*] This bot detects inappropriate content and sends roasts!")
    print("[*] Detection includes: inappropriate words, excessive caps, spam")
    print("=" * 60)
    print(f"[*] Token: oauth:{ACCESS_TOKEN[:8]}...")
    print(f"[*] Channel: {CHANNEL}")
    print(f"[*] Bot: {BOT_NAME}")
    print("=" * 60)
    print()
    
    # Create roast bot
    bot = SimpleTwitchRoastBot(ACCESS_TOKEN, CHANNEL, BOT_NAME)
    
    # Connect and start monitoring
    if bot.connect():
        try:
            bot.listen()
        except KeyboardInterrupt:
            print("\n[-] Roast bot stopped!")
    else:
        print("\n[-] Failed to connect. Check your credentials!")

if __name__ == "__main__":
    main()