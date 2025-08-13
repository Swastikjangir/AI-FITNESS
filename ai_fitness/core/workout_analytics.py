"""
Workout Analytics module for AI Fitness Coach.

This module provides workout tracking and data management capabilities.
"""

import pandas as pd
from datetime import datetime, timedelta
import os

from ai_fitness.config.settings import get_settings

class WorkoutAnalytics:
    """Workout analytics and progress tracking class"""
    
    def __init__(self, log_file=None):
        self.settings = get_settings()
        self.log_file = log_file or self.settings.workout_log_file
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load workout data from CSV"""
        if os.path.exists(self.log_file):
            self.df = pd.read_csv(self.log_file)
            # Ensure timestamp is properly formatted
            if 'timestamp' in self.df.columns:
                self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
                self.df['date'] = self.df['timestamp'].dt.date
            return self.df
        else:
            print(f"No workout data found at {self.log_file}")
            self.df = pd.DataFrame()
            return self.df
    
    def get_daily_summary(self):
        """Get daily exercise summary"""
        if self.df is None or self.df.empty:
            return pd.DataFrame()
        
        # Check if date column exists, if not create it from timestamp
        if 'date' not in self.df.columns and 'timestamp' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['timestamp']).dt.date
        
        # If still no date column, return empty DataFrame
        if 'date' not in self.df.columns:
            return pd.DataFrame()
        
        daily_summary = self.df.groupby(['date', 'exercise'])['count'].sum().reset_index()
        return daily_summary
    
    def get_weekly_progress(self):
        """Get weekly progress data"""
        if self.df is None or self.df.empty:
            return pd.DataFrame()
        
        # Check if timestamp column exists
        if 'timestamp' not in self.df.columns:
            return pd.DataFrame()
        
        # Create date column if it doesn't exist
        if 'date' not in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['timestamp']).dt.date
        
        self.df['week'] = self.df['timestamp'].dt.isocalendar().week
        self.df['year'] = self.df['timestamp'].dt.year
        weekly_progress = self.df.groupby(['year', 'week', 'exercise'])['count'].sum().reset_index()
        return weekly_progress
    
    def save_workout_data(self, exercise: str, count: int, duration: int = 0, calories: int = 0):
        """Save a single workout session to the database"""
        from datetime import datetime
        
        workout_data = {
            'timestamp': datetime.now(),
            'exercise': exercise,
            'count': count,
            'duration': duration,
            'calories': calories
        }
        
        # Add to DataFrame
        if self.df is None:
            self.df = pd.DataFrame()
        
        # Convert to DataFrame and ensure timestamp is datetime
        new_row = pd.DataFrame([workout_data])
        new_row['timestamp'] = pd.to_datetime(new_row['timestamp'])
        
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        
        # Ensure timestamp column is datetime
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        # Save to CSV
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self.df.to_csv(self.log_file, index=False)
        
        print(f"Workout data saved: {exercise} - {count} reps")
        return workout_data

    def get_workout_summary(self, pushup_count: int, squat_count: int, duration: int):
        """Get a summary of the current workout session"""
        total_reps = pushup_count + squat_count
        total_calories = (pushup_count * 0.5) + (squat_count * 0.3)
        
        summary = {
            'total_reps': total_reps,
            'push_ups': pushup_count,
            'squats': squat_count,
            'duration_seconds': duration,
            'duration_minutes': round(duration / 60, 1),
            'total_calories': round(total_calories, 1),
            'pushup_calories': round(pushup_count * 0.5, 1),
            'squat_calories': round(squat_count * 0.3, 1),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return summary

    def create_sample_workout_data(self):
        """Create sample workout data for testing and demonstration"""
        from datetime import datetime, timedelta
        import random
        
        # Generate sample data for the last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Sample exercises
        exercises = ['push_ups', 'squats', 'pull_ups', 'lunges', 'planks']
        
        data = []
        current_date = start_date
        
        while current_date <= end_date:
            # Randomly decide if there's a workout on this day (70% chance)
            if random.random() < 0.7:
                # Random number of exercises per day (1-3)
                num_exercises = random.randint(1, 3)
                selected_exercises = random.sample(exercises, num_exercises)
                
                for exercise in selected_exercises:
                    # Random number of reps (5-50)
                    reps = random.randint(5, 50)
                    # Random time during the day
                    workout_time = current_date.replace(
                        hour=random.randint(6, 22),
                        minute=random.randint(0, 59)
                    )
                    
                    data.append({
                        'timestamp': workout_time,
                        'exercise': exercise,
                        'count': reps,
                        'duration': random.randint(30, 300),  # 30 seconds to 5 minutes
                        'calories': random.randint(10, 100)
                    })
            
            current_date += timedelta(days=1)
        
        # Create DataFrame
        self.df = pd.DataFrame(data)
        
        # Ensure timestamp is properly formatted
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            self.df['date'] = self.df['timestamp'].dt.date
        
        # Save to CSV
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self.df.to_csv(self.log_file, index=False)
        
        print(f"Sample workout data created with {len(data)} records")
        return self.df

    def generate_workout_report(self):
        """Generate a comprehensive workout report"""
        if self.df is None or self.df.empty:
            print("No workout data available for report generation")
            return
        
        print("=" * 50)
        print("AI FITNESS COACH - WORKOUT REPORT")
        print("=" * 50)
        
        # Overall statistics
        total_sessions = len(self.df['date'].unique())
        total_reps = self.df['count'].sum()
        exercises_done = self.df['exercise'].nunique()
        
        print(f"Total Workout Sessions: {total_sessions}")
        print(f"Total Reps Completed: {total_reps}")
        print(f"Different Exercises: {exercises_done}")
        print()
        
        # Exercise breakdown
        print("EXERCISE BREAKDOWN:")
        exercise_summary = self.df.groupby('exercise')['count'].agg(['sum', 'mean', 'count']).round(2)
        exercise_summary.columns = ['Total Reps', 'Avg Reps per Session', 'Sessions']
        print(exercise_summary)
        print()
        
        # Recent activity
        print("RECENT ACTIVITY (Last 7 days):")
        recent_date = datetime.now().date() - timedelta(days=7)
        recent_data = self.df[self.df['date'] >= recent_date]
        if not recent_data.empty:
            recent_summary = recent_data.groupby('exercise')['count'].sum()
            for exercise, count in recent_summary.items():
                print(f"  {exercise.title()}: {count} reps")
        else:
            print("  No recent activity")
        
        print("=" * 50)

if __name__ == "__main__":
    # Example usage
    analytics = WorkoutAnalytics()
    analytics.generate_workout_report()
