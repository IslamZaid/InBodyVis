"""
Database module for Body Composition Dashboard.
Uses SQLite for persistent storage of user data and reports.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Optional, Dict, List, Any
import pandas as pd
import bcrypt


# Database file path - stored in app directory
DB_PATH = os.path.join(os.path.dirname(__file__), 'inbodyvis.db')


def get_connection():
    """Get a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Initialize the database with required tables."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Reports table - stores extracted PDF data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            report_date TIMESTAMP,
            data_json TEXT NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()


# ==============================================================================
# User Management Functions
# ==============================================================================

def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash."""
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except:
        return False


def create_user(username: str, email: str, password: str, name: str) -> tuple[bool, str]:
    """
    Create a new user.
    Returns (success, message) tuple.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Check if username exists
        cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
        if cursor.fetchone():
            return False, "Username already exists"
        
        # Check if email exists
        cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
        if cursor.fetchone():
            return False, "Email already registered"
        
        # Create user
        password_hash = hash_password(password)
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, name)
            VALUES (?, ?, ?, ?)
        ''', (username, email, password_hash, name))
        
        conn.commit()
        return True, "Account created successfully!"
        
    except Exception as e:
        return False, f"Error creating account: {str(e)}"
    finally:
        conn.close()


def authenticate_user(username: str, password: str) -> tuple[bool, Optional[Dict]]:
    """
    Authenticate a user.
    Returns (success, user_info) tuple.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            SELECT id, username, email, password_hash, name
            FROM users WHERE username = ?
        ''', (username,))
        
        row = cursor.fetchone()
        if not row:
            return False, None
        
        if verify_password(password, row['password_hash']):
            return True, {
                'id': row['id'],
                'username': row['username'],
                'email': row['email'],
                'name': row['name']
            }
        return False, None
        
    finally:
        conn.close()


def get_user_by_username(username: str) -> Optional[Dict]:
    """Get user info by username."""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            SELECT id, username, email, name
            FROM users WHERE username = ?
        ''', (username,))
        
        row = cursor.fetchone()
        if row:
            return {
                'id': row['id'],
                'username': row['username'],
                'email': row['email'],
                'name': row['name']
            }
        return None
    finally:
        conn.close()


# ==============================================================================
# Report Management Functions
# ==============================================================================

def save_report(user_id: int, filename: str, data: Dict, report_date: datetime = None) -> bool:
    """Save an extracted report to the database."""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        data_json = json.dumps(data)
        cursor.execute('''
            INSERT INTO reports (user_id, filename, report_date, data_json)
            VALUES (?, ?, ?, ?)
        ''', (user_id, filename, report_date, data_json))
        
        conn.commit()
        return True
    except Exception as e:
        print(f"Error saving report: {e}")
        return False
    finally:
        conn.close()


def get_user_reports(user_id: int) -> List[Dict]:
    """Get all reports for a user."""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            SELECT id, filename, report_date, data_json, uploaded_at
            FROM reports 
            WHERE user_id = ?
            ORDER BY report_date DESC
        ''', (user_id,))
        
        reports = []
        for row in cursor.fetchall():
            reports.append({
                'id': row['id'],
                'filename': row['filename'],
                'report_date': row['report_date'],
                'data': json.loads(row['data_json']),
                'uploaded_at': row['uploaded_at']
            })
        return reports
        
    finally:
        conn.close()


def get_user_reports_dataframe(user_id: int) -> pd.DataFrame:
    """Get all user reports as a pandas DataFrame."""
    reports = get_user_reports(user_id)
    
    if not reports:
        return pd.DataFrame()
    
    # Extract data from each report
    all_data = []
    for report in reports:
        data = report['data'].copy()
        data['Source File'] = report['filename']
        data['Uploaded At'] = report['uploaded_at']
        all_data.append(data)
    
    return pd.DataFrame(all_data)


def delete_report(report_id: int, user_id: int) -> bool:
    """Delete a report (only if owned by user)."""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            DELETE FROM reports 
            WHERE id = ? AND user_id = ?
        ''', (report_id, user_id))
        
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def get_report_count(user_id: int) -> int:
    """Get the number of reports for a user."""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('SELECT COUNT(*) FROM reports WHERE user_id = ?', (user_id,))
        return cursor.fetchone()[0]
    finally:
        conn.close()


# Initialize database on module load
init_database()
