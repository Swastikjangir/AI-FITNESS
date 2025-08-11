"""
Data Processing module for AI Fitness Coach.

This module handles data operations including data cleaning, validation,
transformation, and storage for workout and fitness data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

from ai_fitness.config.settings import get_settings

class DataProcessor:
    """Data processing and management class for fitness data"""
    
    def __init__(self):
        self.settings = get_settings()
        self.data_dir = self.settings.data_dir
        self.workout_log_file = self.settings.workout_log_file
        
    def create_sample_workout_data(self) -> pd.DataFrame:
        """Create sample workout data for testing and demonstration"""
        # Generate sample data for the last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Sample exercises
        exercises = ['push_ups', 'squats', 'pull_ups', 'lunges', 'planks']
        
        data = []
        current_date = start_date
        
        while current_date <= end_date:
            # Randomly select 2-4 exercises per day
            num_exercises = np.random.randint(2, 5)
            daily_exercises = np.random.choice(exercises, num_exercises, replace=False)
            
            for exercise in daily_exercises:
                # Generate realistic rep counts
                if exercise in ['push_ups', 'squats']:
                    reps = np.random.randint(10, 51)
                elif exercise in ['pull_ups']:
                    reps = np.random.randint(5, 21)
                elif exercise in ['lunges']:
                    reps = np.random.randint(15, 31)
                else:  # planks
                    reps = np.random.randint(30, 181)  # seconds
                
                data.append({
                    'timestamp': current_date.strftime('%Y-%m-%d %H:%M:%S'),
                    'exercise': exercise,
                    'count': reps,
                    'duration': np.random.randint(30, 120),  # minutes
                    'intensity': np.random.choice(['low', 'medium', 'high']),
                    'notes': f'Sample {exercise} workout'
                })
            
            current_date += timedelta(days=1)
        
        return pd.DataFrame(data)
    
    def save_workout_data(self, data: Union[pd.DataFrame, List[Dict]], filename: str = None) -> bool:
        """Save workout data to CSV file"""
        try:
            if filename is None:
                filename = self.workout_log_file
            
            # Ensure data directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data
            
            # Add timestamp if not present
            if 'timestamp' not in df.columns:
                df['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            df.to_csv(filename, index=False)
            print(f"Workout data saved to {filename}")
            return True
            
        except Exception as e:
            print(f"Error saving workout data: {str(e)}")
            return False
    
    def load_workout_data(self, filename: str = None) -> pd.DataFrame:
        """Load workout data from CSV file"""
        try:
            if filename is None:
                filename = self.workout_log_file
            
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                # Convert timestamp to datetime
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
            else:
                print(f"No workout data file found at {filename}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error loading workout data: {str(e)}")
            return pd.DataFrame()
    
    def validate_workout_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate workout data for consistency and completeness"""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'total_records': len(data),
            'missing_values': {},
            'data_types': {},
            'value_ranges': {}
        }
        
        if data.empty:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Data is empty")
            return validation_results
        
        # Check for required columns
        required_columns = ['timestamp', 'exercise', 'count']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_columns}")
        
        # Check for missing values
        for col in data.columns:
            missing_count = data[col].isnull().sum()
            if missing_count > 0:
                validation_results['missing_values'][col] = missing_count
                if col in required_columns:
                    validation_results['is_valid'] = False
                    validation_results['errors'].append(f"Missing values in required column: {col}")
                else:
                    validation_results['warnings'].append(f"Missing values in optional column: {col}")
        
        # Check data types
        for col in data.columns:
            validation_results['data_types'][col] = str(data[col].dtype)
        
        # Check value ranges
        if 'count' in data.columns:
            count_stats = data['count'].describe()
            validation_results['value_ranges']['count'] = {
                'min': count_stats['min'],
                'max': count_stats['max'],
                'mean': count_stats['mean']
            }
            
            # Check for unrealistic values
            if count_stats['max'] > 1000:
                validation_results['warnings'].append("Some count values seem unusually high")
            if count_stats['min'] < 0:
                validation_results['is_valid'] = False
                validation_results['errors'].append("Negative count values found")
        
        return validation_results
    
    def clean_workout_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess workout data"""
        cleaned_data = data.copy()
        
        # Remove duplicate rows
        initial_rows = len(cleaned_data)
        cleaned_data = cleaned_data.drop_duplicates()
        if len(cleaned_data) < initial_rows:
            print(f"Removed {initial_rows - len(cleaned_data)} duplicate rows")
        
        # Handle missing values
        if 'notes' in cleaned_data.columns:
            cleaned_data['notes'] = cleaned_data['notes'].fillna('')
        
        # Remove rows with missing required values
        required_columns = ['timestamp', 'exercise', 'count']
        for col in required_columns:
            if col in cleaned_data.columns:
                cleaned_data = cleaned_data.dropna(subset=[col])
        
        # Convert timestamp to datetime if it's not already
        if 'timestamp' in cleaned_data.columns:
            if cleaned_data['timestamp'].dtype == 'object':
                cleaned_data['timestamp'] = pd.to_datetime(cleaned_data['timestamp'], errors='coerce')
                # Remove rows with invalid timestamps
                cleaned_data = cleaned_data.dropna(subset=['timestamp'])
        
        # Ensure count values are numeric and positive
        if 'count' in cleaned_data.columns:
            cleaned_data['count'] = pd.to_numeric(cleaned_data['count'], errors='coerce')
            cleaned_data = cleaned_data.dropna(subset=['count'])
            cleaned_data = cleaned_data[cleaned_data['count'] > 0]
        
        # Sort by timestamp
        if 'timestamp' in cleaned_data.columns:
            cleaned_data = cleaned_data.sort_values('timestamp')
        
        return cleaned_data
    
    def aggregate_workout_data(self, data: pd.DataFrame, group_by: str = 'date') -> pd.DataFrame:
        """Aggregate workout data by specified grouping"""
        if data.empty:
            return pd.DataFrame()
        
        # Ensure timestamp column exists and is datetime
        if 'timestamp' not in data.columns:
            print("No timestamp column found for aggregation")
            return data
        
        if data['timestamp'].dtype != 'datetime64[ns]':
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Add date column if grouping by date
        if group_by == 'date':
            data['date'] = data['timestamp'].dt.date
        
        # Group and aggregate
        if group_by == 'date':
            aggregated = data.groupby(['date', 'exercise']).agg({
                'count': ['sum', 'mean', 'count'],
                'duration': 'sum' if 'duration' in data.columns else 'count'
            }).reset_index()
            
            # Flatten column names
            aggregated.columns = ['date', 'exercise', 'total_count', 'avg_count', 'sessions', 'total_duration']
            
        elif group_by == 'week':
            data['week'] = data['timestamp'].dt.isocalendar().week
            data['year'] = data['timestamp'].dt.year
            
            aggregated = data.groupby(['year', 'week', 'exercise']).agg({
                'count': ['sum', 'mean', 'count'],
                'duration': 'sum' if 'duration' in data.columns else 'count'
            }).reset_index()
            
            aggregated.columns = ['year', 'week', 'exercise', 'total_count', 'avg_count', 'sessions', 'total_duration']
            
        elif group_by == 'month':
            data['month'] = data['timestamp'].dt.month
            data['year'] = data['timestamp'].dt.year
            
            aggregated = data.groupby(['year', 'month', 'exercise']).agg({
                'count': ['sum', 'mean', 'count'],
                'duration': 'sum' if 'duration' in data.columns else 'count'
            }).reset_index()
            
            aggregated.columns = ['year', 'month', 'exercise', 'total_count', 'avg_count', 'sessions', 'total_duration']
        
        else:
            print(f"Unknown grouping option: {group_by}")
            return data
        
        return aggregated
    
    def export_data(self, data: pd.DataFrame, format: str = 'csv', filename: str = None) -> bool:
        """Export data in various formats"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = self.data_dir / f"workout_data_{timestamp}.{format}"
            
            if format.lower() == 'csv':
                data.to_csv(filename, index=False)
            elif format.lower() == 'json':
                data.to_json(filename, orient='records', indent=2)
            elif format.lower() == 'excel':
                data.to_excel(filename, index=False)
            else:
                print(f"Unsupported format: {format}")
                return False
            
            print(f"Data exported to {filename}")
            return True
            
        except Exception as e:
            print(f"Error exporting data: {str(e)}")
            return False

if __name__ == "__main__":
    # Example usage
    processor = DataProcessor()
    
    # Create sample data
    sample_data = processor.create_sample_workout_data()
    print(f"Created sample data with {len(sample_data)} records")
    
    # Save sample data
    processor.save_workout_data(sample_data)
    
    # Load and validate data
    loaded_data = processor.load_workout_data()
    validation = processor.validate_workout_data(loaded_data)
    print(f"Data validation: {validation['is_valid']}")
    
    # Clean data
    cleaned_data = processor.clean_workout_data(loaded_data)
    print(f"Cleaned data has {len(cleaned_data)} records")
    
    # Aggregate data
    daily_summary = processor.aggregate_workout_data(cleaned_data, 'date')
    print(f"Daily summary has {len(daily_summary)} records")
