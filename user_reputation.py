"""
User reputation tracking module using SQLite for persistent storage.
Tracks nontoxic_count, toxic_count, total_messages, and last_seen for each user.
"""

import sqlite3
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path


DB_FILE = "user_reputation.db"


def init_database(db_path: str = DB_FILE) -> None:
    """
    Initialize the SQLite database and create the users table if it doesn't exist.
    
    Args:
        db_path: Path to the SQLite database file
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_reputation (
            username TEXT PRIMARY KEY,
            nontoxic_count INTEGER DEFAULT 0,
            toxic_count INTEGER DEFAULT 0,
            total_messages INTEGER DEFAULT 0,
            last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()
    print(f"User reputation database initialized at {db_path}")


def get_user_reputation(username: str, db_path: str = DB_FILE) -> Dict:
    """
    Get the reputation data for a specific user.
    
    Args:
        username: Twitch username (case-insensitive)
        db_path: Path to the SQLite database file
        
    Returns:
        Dictionary with nontoxic_count, toxic_count, total_messages, last_seen
        Returns default values (all 0) if user doesn't exist
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Case-insensitive lookup
    cursor.execute("""
        SELECT nontoxic_count, toxic_count, total_messages, last_seen
        FROM user_reputation
        WHERE LOWER(username) = LOWER(?)
    """, (username,))
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return {
            'username': username,
            'nontoxic_count': result[0],
            'toxic_count': result[1],
            'total_messages': result[2],
            'last_seen': result[3]
        }
    else:
        return {
            'username': username,
            'nontoxic_count': 0,
            'toxic_count': 0,
            'total_messages': 0,
            'last_seen': None
        }


def increment_nontoxic_count(username: str, db_path: str = DB_FILE) -> None:
    """
    Increment the nontoxic message count for a user.
    Creates user record if it doesn't exist.
    
    Args:
        username: Twitch username
        db_path: Path to the SQLite database file
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO user_reputation (username, nontoxic_count, total_messages, last_seen)
        VALUES (?, 1, 1, ?)
        ON CONFLICT(username) DO UPDATE SET
            nontoxic_count = nontoxic_count + 1,
            total_messages = total_messages + 1,
            last_seen = ?
    """, (username, datetime.now(), datetime.now()))
    
    conn.commit()
    conn.close()


def record_toxic_behavior(username: str, reduction_amount: int = 10, db_path: str = DB_FILE) -> None:
    """
    Record toxic behavior by incrementing toxic_count and reducing nontoxic_count.
    
    Args:
        username: Twitch username
        reduction_amount: Amount to reduce nontoxic_count by (default 10, minimum 0)
        db_path: Path to the SQLite database file
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO user_reputation (username, toxic_count, total_messages, nontoxic_count, last_seen)
        VALUES (?, 1, 1, 0, ?)
        ON CONFLICT(username) DO UPDATE SET
            toxic_count = toxic_count + 1,
            total_messages = total_messages + 1,
            nontoxic_count = MAX(0, nontoxic_count - ?),
            last_seen = ?
    """, (username, datetime.now(), reduction_amount, datetime.now()))
    
    conn.commit()
    conn.close()


def reduce_toxic_count(username: str, amount: int = 1, db_path: str = DB_FILE) -> bool:
    """
    Reduce the toxic count for a user (for manual moderation/forgiveness).
    Used by the !notoxic bot command.
    
    Args:
        username: Twitch username
        amount: Amount to reduce toxic_count by (default 1)
        db_path: Path to the SQLite database file
        
    Returns:
        True if user exists and was updated, False otherwise
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if user exists
    cursor.execute("""
        SELECT toxic_count FROM user_reputation
        WHERE LOWER(username) = LOWER(?)
    """, (username,))
    
    result = cursor.fetchone()
    
    if result is None:
        conn.close()
        return False
    
    # Update toxic count (minimum 0)
    cursor.execute("""
        UPDATE user_reputation
        SET toxic_count = MAX(0, toxic_count - ?)
        WHERE LOWER(username) = LOWER(?)
    """, (amount, username))
    
    conn.commit()
    conn.close()
    return True


def cleanup_old_users(days_inactive: int = 90, db_path: str = DB_FILE) -> int:
    """
    Delete user records that haven't been seen in X days.
    
    Args:
        days_inactive: Number of days of inactivity before deletion
        db_path: Path to the SQLite database file
        
    Returns:
        Number of users deleted
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        DELETE FROM user_reputation
        WHERE last_seen < datetime('now', '-' || ? || ' days')
    """, (days_inactive,))
    
    deleted_count = cursor.rowcount
    conn.commit()
    conn.close()
    
    return deleted_count


def get_reputation_stats(db_path: str = DB_FILE) -> Dict:
    """
    Get overall statistics about the reputation database.
    
    Args:
        db_path: Path to the SQLite database file
        
    Returns:
        Dictionary with total_users, avg_nontoxic_count, avg_toxic_count, etc.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            COUNT(*) as total_users,
            AVG(nontoxic_count) as avg_nontoxic,
            AVG(toxic_count) as avg_toxic,
            SUM(nontoxic_count) as total_nontoxic,
            SUM(toxic_count) as total_toxic,
            SUM(total_messages) as total_messages
        FROM user_reputation
    """)
    
    result = cursor.fetchone()
    conn.close()
    
    return {
        'total_users': result[0] or 0,
        'avg_nontoxic_count': round(result[1] or 0, 2),
        'avg_toxic_count': round(result[2] or 0, 2),
        'total_nontoxic_messages': result[3] or 0,
        'total_toxic_messages': result[4] or 0,
        'total_messages': result[5] or 0
    }


if __name__ == "__main__":
    # Test the database functions
    print("Testing user reputation database...")
    
    # Initialize database
    test_db = "test_reputation.db"
    init_database(test_db)
    
    # Test increment nontoxic
    print("\nAdding nontoxic messages for user 'testuser'...")
    for i in range(5):
        increment_nontoxic_count("testuser", test_db)
    
    # Check reputation
    rep = get_user_reputation("testuser", test_db)
    print(f"User reputation: {rep}")
    
    # Record toxic behavior
    print("\nRecording toxic behavior...")
    record_toxic_behavior("testuser", reduction_amount=2, db_path=test_db)
    
    # Check reputation again
    rep = get_user_reputation("testuser", test_db)
    print(f"User reputation after toxic: {rep}")
    
    # Test reduce toxic count
    print("\nReducing toxic count by 1...")
    reduce_toxic_count("testuser", amount=1, db_path=test_db)
    rep = get_user_reputation("testuser", test_db)
    print(f"User reputation after reduction: {rep}")
    
    # Get stats
    stats = get_reputation_stats(test_db)
    print(f"\nDatabase stats: {stats}")
    
    # Cleanup
    Path(test_db).unlink()
    print(f"\nTest database deleted.")
