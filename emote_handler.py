"""
Emote handling module for Twitch bot.
Handles fetching emotes from 7TV, BTTV, FFZ and stripping them from messages.
"""

import re
import requests
from datetime import datetime, timedelta

# Emote cache variables
emote_list_cache = None
emote_cache_time = None
all_emote_names = set()  # Combined set of all emote names for stripping
emote_names_cache_time = None
EMOTE_CACHE_DURATION = timedelta(seconds=10)  # Refresh every 10 seconds


def fetch_7tv_emotes(user_id: str) -> str:
    """
    Fetch emotes from 7TV GraphQL API and format them for the LLM.
    
    Args:
        user_id: The 7TV user ID to fetch emotes for
        
    Returns:
        Formatted string of emote names for LLM context
    """
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
                'variables': {'userId': user_id}
            },
            headers={'Content-Type': 'application/json'}
        )
        response.raise_for_status()
        data = response.json()
        
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


def fetch_all_emote_names(user_id: str) -> set:
    """
    Fetch all emote names from 7TV (global + user), BTTV (global), and FFZ (global) for stripping.
    
    Args:
        user_id: The 7TV user ID to fetch emotes for
        
    Returns:
        Set of all emote names
    """
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
                'variables': {'userId': user_id}
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


def strip_twitch_emotes(message_text: str, emotes) -> str:
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


def strip_third_party_emotes(message_text: str, emote_names: set) -> str:
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


def strip_all_emotes(message_text: str, twitch_emotes, third_party_emote_names: set) -> str:
    """
    Remove all emotes (Twitch + third-party) from a message.
    
    Args:
        message_text: The raw message text
        twitch_emotes: Twitch emote data from ChatMessage.emotes
        third_party_emote_names: Set of third-party emote names to remove
    
    Returns:
        Message text with all emotes removed
    """
    # First strip Twitch emotes using position data
    text = strip_twitch_emotes(message_text, twitch_emotes)
    
    # Then strip third-party emotes using name matching
    text = strip_third_party_emotes(text, third_party_emote_names)
    
    return text


def get_all_emote_names() -> set:
    """
    Get the cached set of all emote names.
    
    Returns:
        Set of all emote names
    """
    return all_emote_names
