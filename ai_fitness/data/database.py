"""
Database management module for AI Fitness Coach.

This module provides database connectivity and management for storing
workout data, user profiles, and fitness analytics.
"""

import sqlite3
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
import os
from pathlib import Path

from ai_fitness.config.settings import get_settings

class DatabaseManager:
    """Database management class for fitness data"""
    
    def __init__(self, db_path: str = None):
        self.settings = get_settings()
        self.db_path = db_path or self.settings.database_url.replace('sqlite:///', '')
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize database
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create workout_logs table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS workout_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                        exercise TEXT NOT NULL,
                        count INTEGER NOT NULL,
                        duration INTEGER,
                        intensity TEXT,
                        notes TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create user_profiles table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_profiles (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        age INTEGER,
                        gender TEXT,
                        height REAL,
                        weight REAL,
                        body_type TEXT,
                        fitness_level TEXT,
                        goals TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create fitness_goals table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS fitness_goals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER,
                        goal_type TEXT NOT NULL,
                        target_value REAL,
                        current_value REAL,
                        target_date DATE,
                        status TEXT DEFAULT 'active',
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES user_profiles (id)
                    )
                ''')
                
                # Create workout_plans table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS workout_plans (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER,
                        plan_name TEXT NOT NULL,
                        plan_data TEXT NOT NULL,
                        is_active BOOLEAN DEFAULT 1,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES user_profiles (id)
                    )
                ''')
                
                conn.commit()
                print("Database initialized successfully")
                
        except Exception as e:
            print(f"Error initializing database: {str(e)}")
    
    def add_workout_log(self, workout_data: Dict[str, Any]) -> bool:
        """Add a new workout log entry"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO workout_logs (timestamp, exercise, count, duration, intensity, notes)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    workout_data.get('timestamp', datetime.now().isoformat()),
                    workout_data.get('exercise'),
                    workout_data.get('count'),
                    workout_data.get('duration'),
                    workout_data.get('intensity'),
                    workout_data.get('notes', '')
                ))
                
                conn.commit()
                print(f"Workout log added: {workout_data.get('exercise')}")
                return True
                
        except Exception as e:
            print(f"Error adding workout log: {str(e)}")
            return False
    
    def get_workout_logs(self, limit: int = 100, offset: int = 0, 
                         exercise: str = None, date_from: str = None, 
                         date_to: str = None) -> pd.DataFrame:
        """Get workout logs with optional filtering"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM workout_logs WHERE 1=1"
                params = []
                
                if exercise:
                    query += " AND exercise = ?"
                    params.append(exercise)
                
                if date_from:
                    query += " AND timestamp >= ?"
                    params.append(date_from)
                
                if date_to:
                    query += " AND timestamp <= ?"
                    params.append(date_to)
                
                query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                
                df = pd.read_sql_query(query, conn, params=params)
                
                if not df.empty and 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                return df
                
        except Exception as e:
            print(f"Error getting workout logs: {str(e)}")
            return pd.DataFrame()
    
    def add_user_profile(self, profile_data: Dict[str, Any]) -> int:
        """Add a new user profile"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO user_profiles (name, age, gender, height, weight, body_type, fitness_level, goals)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    profile_data.get('name'),
                    profile_data.get('age'),
                    profile_data.get('gender'),
                    profile_data.get('height'),
                    profile_data.get('weight'),
                    profile_data.get('body_type'),
                    profile_data.get('fitness_level'),
                    json.dumps(profile_data.get('goals', []))
                ))
                
                user_id = cursor.lastrowid
                conn.commit()
                print(f"User profile added with ID: {user_id}")
                return user_id
                
        except Exception as e:
            print(f"Error adding user profile: {str(e)}")
            return -1
    
    def get_user_profile(self, user_id: int) -> Dict[str, Any]:
        """Get user profile by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM user_profiles WHERE id = ?
                ''', (user_id,))
                
                row = cursor.fetchone()
                if row:
                    columns = [description[0] for description in cursor.description]
                    profile = dict(zip(columns, row))
                    
                    # Parse goals JSON
                    if profile.get('goals'):
                        try:
                            profile['goals'] = json.loads(profile['goals'])
                        except:
                            profile['goals'] = []
                    
                    return profile
                else:
                    return {}
                    
        except Exception as e:
            print(f"Error getting user profile: {str(e)}")
            return {}
    
    def update_user_profile(self, user_id: int, profile_data: Dict[str, Any]) -> bool:
        """Update user profile"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build dynamic update query
                update_fields = []
                params = []
                
                for key, value in profile_data.items():
                    if key in ['name', 'age', 'gender', 'height', 'weight', 'body_type', 'fitness_level', 'goals']:
                        update_fields.append(f"{key} = ?")
                        if key == 'goals':
                            params.append(json.dumps(value))
                        else:
                            params.append(value)
                
                if not update_fields:
                    print("No valid fields to update")
                    return False
                
                update_fields.append("updated_at = CURRENT_TIMESTAMP")
                params.append(user_id)
                
                query = f"UPDATE user_profiles SET {', '.join(update_fields)} WHERE id = ?"
                cursor.execute(query, params)
                
                conn.commit()
                print(f"User profile {user_id} updated successfully")
                return True
                
        except Exception as e:
            print(f"Error updating user profile: {str(e)}")
            return False
    
    def add_fitness_goal(self, user_id: int, goal_data: Dict[str, Any]) -> int:
        """Add a new fitness goal"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO fitness_goals (user_id, goal_type, target_value, current_value, target_date)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    user_id,
                    goal_data.get('goal_type'),
                    goal_data.get('target_value'),
                    goal_data.get('current_value', 0),
                    goal_data.get('target_date')
                ))
                
                goal_id = cursor.lastrowid
                conn.commit()
                print(f"Fitness goal added with ID: {goal_id}")
                return goal_id
                
        except Exception as e:
            print(f"Error adding fitness goal: {str(e)}")
            return -1
    
    def get_fitness_goals(self, user_id: int) -> pd.DataFrame:
        """Get fitness goals for a user"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT * FROM fitness_goals 
                    WHERE user_id = ? AND status = 'active'
                    ORDER BY target_date ASC
                '''
                
                df = pd.read_sql_query(query, conn, params=[user_id])
                return df
                
        except Exception as e:
            print(f"Error getting fitness goals: {str(e)}")
            return pd.DataFrame()
    
    def save_workout_plan(self, user_id: int, plan_name: str, plan_data: Dict[str, Any]) -> int:
        """Save a workout plan"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO workout_plans (user_id, plan_name, plan_data)
                    VALUES (?, ?, ?)
                ''', (
                    user_id,
                    plan_name,
                    json.dumps(plan_data)
                ))
                
                plan_id = cursor.lastrowid
                conn.commit()
                print(f"Workout plan saved with ID: {plan_id}")
                return plan_id
                
        except Exception as e:
            print(f"Error saving workout plan: {str(e)}")
            return -1
    
    def get_workout_plans(self, user_id: int) -> pd.DataFrame:
        """Get workout plans for a user"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT * FROM workout_plans 
                    WHERE user_id = ? AND is_active = 1
                    ORDER BY created_at DESC
                '''
                
                df = pd.read_sql_query(query, conn, params=[user_id])
                
                # Parse plan_data JSON
                if not df.empty and 'plan_data' in df.columns:
                    df['plan_data'] = df['plan_data'].apply(lambda x: json.loads(x) if x else {})
                
                return df
                
        except Exception as e:
            print(f"Error getting workout plans: {str(e)}")
            return pd.DataFrame()
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Count records in each table
                tables = ['workout_logs', 'user_profiles', 'fitness_goals', 'workout_plans']
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[f"{table}_count"] = cursor.fetchone()[0]
                
                # Get database size
                cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
                stats['database_size_bytes'] = cursor.fetchone()[0]
                
                return stats
                
        except Exception as e:
            print(f"Error getting database stats: {str(e)}")
            return {}
    
    def backup_database(self, backup_path: str = None) -> bool:
        """Create a backup of the database"""
        try:
            if backup_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = f"backup_ai_fitness_{timestamp}.db"
            
            with sqlite3.connect(self.db_path) as source_conn:
                with sqlite3.connect(backup_path) as backup_conn:
                    source_conn.backup(backup_conn)
            
            print(f"Database backed up to: {backup_path}")
            return True
            
        except Exception as e:
            print(f"Error backing up database: {str(e)}")
            return False

if __name__ == "__main__":
    # Example usage
    db_manager = DatabaseManager()
    
    # Add sample workout log
    workout_data = {
        'exercise': 'push_ups',
        'count': 25,
        'duration': 45,
        'intensity': 'medium',
        'notes': 'Morning workout'
    }
    
    db_manager.add_workout_log(workout_data)
    
    # Get workout logs
    logs = db_manager.get_workout_logs(limit=10)
    print(f"Retrieved {len(logs)} workout logs")
    
    # Get database stats
    stats = db_manager.get_database_stats()
    print("Database stats:", stats)
