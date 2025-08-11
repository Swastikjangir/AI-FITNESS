"""
Workout Analytics module for AI Fitness Coach.

This module provides comprehensive workout tracking, analysis, and
visualization capabilities for fitness progress monitoring.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
from pathlib import Path

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
        
        daily_summary = self.df.groupby(['date', 'exercise'])['count'].sum().reset_index()
        return daily_summary
    
    def get_weekly_progress(self):
        """Get weekly progress data"""
        if self.df is None or self.df.empty:
            return pd.DataFrame()
        
        self.df['week'] = self.df['timestamp'].dt.isocalendar().week
        self.df['year'] = self.df['timestamp'].dt.year
        weekly_progress = self.df.groupby(['year', 'week', 'exercise'])['count'].sum().reset_index()
        return weekly_progress
    
    def plot_daily_activity(self, save_path=None):
        """Plot daily exercise activity"""
        if self.df is None or self.df.empty:
            print("No data to plot")
            return
        
        if save_path is None:
            save_path = self.settings.logs_dir / "daily_activity.png"
        
        daily_summary = self.get_daily_summary()
        
        plt.figure(figsize=(12, 6))
        for exercise in daily_summary['exercise'].unique():
            exercise_data = daily_summary[daily_summary['exercise'] == exercise]
            plt.plot(exercise_data['date'], exercise_data['count'], 
                    marker='o', label=exercise.title(), linewidth=2)
        
        plt.title('Daily Exercise Activity', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Reps Count', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_exercise_comparison(self, save_path=None):
        """Plot comparison between different exercises"""
        if self.df is None or self.df.empty:
            print("No data to plot")
            return
        
        if save_path is None:
            save_path = self.settings.logs_dir / "exercise_comparison.png"
        
        exercise_totals = self.df.groupby('exercise')['count'].sum().reset_index()
        
        plt.figure(figsize=(10, 6))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        bars = plt.bar(exercise_totals['exercise'], exercise_totals['count'], 
                      color=colors[:len(exercise_totals)], alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Total Exercise Performance', fontsize=16, fontweight='bold')
        plt.xlabel('Exercise Type', fontsize=12)
        plt.ylabel('Total Reps', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_weekly_trends(self, save_path=None):
        """Plot weekly trends"""
        if self.df is None or self.df.empty:
            print("No data to plot")
            return
        
        if save_path is None:
            save_path = self.settings.logs_dir / "weekly_trends.png"
        
        weekly_data = self.get_weekly_progress()
        
        plt.figure(figsize=(14, 8))
        for exercise in weekly_data['exercise'].unique():
            exercise_data = weekly_data[weekly_data['exercise'] == exercise]
            plt.plot(exercise_data['week'], exercise_data['count'], 
                    marker='s', label=exercise.title(), linewidth=2)
        
        plt.title('Weekly Exercise Trends', fontsize=16, fontweight='bold')
        plt.xlabel('Week Number', fontsize=12)
        plt.ylabel('Total Reps per Week', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
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
    
    def create_heatmap(self, save_path=None):
        """Create activity heatmap"""
        if self.df is None or self.df.empty:
            print("No data to plot")
            return
        
        if save_path is None:
            save_path = self.settings.logs_dir / "activity_heatmap.png"
        
        # Create pivot table for heatmap
        self.df['weekday'] = self.df['timestamp'].dt.day_name()
        heatmap_data = self.df.groupby(['weekday', 'exercise'])['count'].sum().unstack(fill_value=0)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, fmt='g', cmap='YlOrRd', 
                   cbar_kws={'label': 'Total Reps'})
        plt.title('Weekly Activity Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('Exercise Type', fontsize=12)
        plt.ylabel('Day of Week', fontsize=12)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    # Example usage
    analytics = WorkoutAnalytics()
    analytics.generate_workout_report()
    analytics.plot_daily_activity()
    analytics.plot_exercise_comparison()
    analytics.plot_weekly_trends()
    analytics.create_heatmap()
