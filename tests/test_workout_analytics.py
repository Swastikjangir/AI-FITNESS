"""
Unit tests for Workout Analytics module.

Tests the WorkoutAnalytics class and its methods.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_fitness.core.workout_analytics import WorkoutAnalytics

class TestWorkoutAnalytics(unittest.TestCase):
    """Test cases for WorkoutAnalytics class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create sample workout data
        self.sample_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=30, freq='D'),
            'exercise': ['push_ups', 'squats', 'pull_ups'] * 10,
            'count': np.random.randint(10, 51, 30),
            'duration': np.random.randint(30, 120, 30),
            'intensity': ['low', 'medium', 'high'] * 10
        })
        
        # Mock settings
        with patch('ai_fitness.core.workout_analytics.get_settings') as mock_settings:
            mock_settings.return_value = Mock(
                workout_log_file='test_workout_log.csv'
            )
            self.analytics = WorkoutAnalytics()
    
    def test_init(self):
        """Test WorkoutAnalytics initialization"""
        self.assertIsNotNone(self.analytics)
        self.assertEqual(self.analytics.log_file, 'test_workout_log.csv')
    
    @patch('pandas.read_csv')
    def test_load_data_existing_file(self, mock_read_csv):
        """Test loading data from existing CSV file"""
        # Mock successful CSV read
        mock_read_csv.return_value = self.sample_data
        
        result = self.analytics.load_data()
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 30)
        self.assertIn('timestamp', result.columns)
        self.assertIn('exercise', result.columns)
        self.assertIn('count', result.columns)
    
    @patch('pandas.read_csv')
    def test_load_data_no_file(self, mock_read_csv):
        """Test loading data when file doesn't exist"""
        # Mock file not found
        mock_read_csv.side_effect = FileNotFoundError()
        
        result = self.analytics.load_data()
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)
    
    def test_get_daily_summary(self):
        """Test daily exercise summary generation"""
        # Set up data
        self.analytics.df = self.sample_data
        
        summary = self.analytics.get_daily_summary()
        
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertIn('date', summary.columns)
        self.assertIn('exercise', summary.columns)
        self.assertIn('count', summary.columns)
        
        # Check that summary is grouped by date and exercise
        self.assertGreater(len(summary), 0)
    
    def test_get_daily_summary_empty_data(self):
        """Test daily summary with empty data"""
        self.analytics.df = pd.DataFrame()
        
        summary = self.analytics.get_daily_summary()
        
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertTrue(summary.empty)
    
    def test_get_weekly_progress(self):
        """Test weekly progress data generation"""
        # Set up data
        self.analytics.df = self.sample_data
        
        progress = self.analytics.get_weekly_progress()
        
        self.assertIsInstance(progress, pd.DataFrame)
        self.assertIn('year', progress.columns)
        self.assertIn('week', progress.columns)
        self.assertIn('exercise', progress.columns)
        self.assertIn('count', progress.columns)
        
        # Check that progress is grouped by year, week, and exercise
        self.assertGreater(len(progress), 0)
    
    def test_get_weekly_progress_empty_data(self):
        """Test weekly progress with empty data"""
        self.analytics.df = pd.DataFrame()
        
        progress = self.analytics.get_weekly_progress()
        
        self.assertIsInstance(progress, pd.DataFrame)
        self.assertTrue(progress.empty)
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_plot_daily_activity(self, mock_show, mock_savefig, mock_figure):
        """Test daily activity plotting"""
        # Set up data
        self.analytics.df = self.sample_data
        
        # Mock matplotlib
        mock_fig = Mock()
        mock_figure.return_value = mock_fig
        
        self.analytics.plot_daily_activity()
        
        # Verify that plotting functions were called
        mock_figure.assert_called()
        mock_savefig.assert_called()
        mock_show.assert_called()
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_plot_daily_activity_empty_data(self, mock_show, mock_savefig, mock_figure):
        """Test daily activity plotting with empty data"""
        self.analytics.df = pd.DataFrame()
        
        self.analytics.plot_daily_activity()
        
        # Verify that no plotting was done
        mock_figure.assert_not_called()
        mock_savefig.assert_not_called()
        mock_show.assert_not_called()
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_plot_exercise_comparison(self, mock_show, mock_savefig, mock_figure):
        """Test exercise comparison plotting"""
        # Set up data
        self.analytics.df = self.sample_data
        
        # Mock matplotlib
        mock_fig = Mock()
        mock_figure.return_value = mock_fig
        
        self.analytics.plot_exercise_comparison()
        
        # Verify that plotting functions were called
        mock_figure.assert_called()
        mock_savefig.assert_called()
        mock_show.assert_called()
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_plot_weekly_trends(self, mock_show, mock_savefig, mock_figure):
        """Test weekly trends plotting"""
        # Set up data
        self.analytics.df = self.sample_data
        
        # Mock matplotlib
        mock_fig = Mock()
        mock_figure.return_value = mock_fig
        
        self.analytics.plot_weekly_trends()
        
        # Verify that plotting functions were called
        mock_figure.assert_called()
        mock_savefig.assert_called()
        mock_show.assert_called()
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_create_heatmap(self, mock_show, mock_savefig, mock_figure):
        """Test activity heatmap creation"""
        # Set up data
        self.analytics.df = self.sample_data
        
        # Mock matplotlib and seaborn
        mock_fig = Mock()
        mock_figure.return_value = mock_fig
        
        with patch('seaborn.heatmap'):
            self.analytics.create_heatmap()
        
        # Verify that plotting functions were called
        mock_figure.assert_called()
        mock_savefig.assert_called()
        mock_show.assert_called()
    
    @patch('builtins.print')
    def test_generate_workout_report(self, mock_print):
        """Test workout report generation"""
        # Set up data
        self.analytics.df = self.sample_data
        
        self.analytics.generate_workout_report()
        
        # Verify that print was called multiple times (report generation)
        self.assertGreater(mock_print.call_count, 10)
    
    @patch('builtins.print')
    def test_generate_workout_report_empty_data(self, mock_print):
        """Test workout report generation with empty data"""
        self.analytics.df = pd.DataFrame()
        
        self.analytics.generate_workout_report()
        
        # Verify that appropriate message was printed
        mock_print.assert_called_with("No workout data available for report generation")
    
    def test_custom_save_paths(self):
        """Test plotting with custom save paths"""
        # Set up data
        self.analytics.df = self.sample_data
        
        custom_path = 'custom_path.png'
        
        with patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.show'):
            
            self.analytics.plot_daily_activity(save_path=custom_path)
            
            # Verify custom path was used
            mock_savefig.assert_called_with(custom_path, dpi=300, bbox_inches='tight')
    
    def test_data_validation(self):
        """Test data validation and error handling"""
        # Test with invalid data
        invalid_data = pd.DataFrame({
            'timestamp': ['invalid_date', '2024-01-01'],
            'exercise': ['push_ups', 'squats'],
            'count': ['not_a_number', 25]
        })
        
        self.analytics.df = invalid_data
        
        # These should not raise exceptions
        try:
            summary = self.analytics.get_daily_summary()
            self.assertIsInstance(summary, pd.DataFrame)
        except Exception as e:
            self.fail(f"get_daily_summary raised an exception: {e}")
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Test with single record
        single_record = pd.DataFrame({
            'timestamp': ['2024-01-01'],
            'exercise': ['push_ups'],
            'count': [25]
        })
        
        self.analytics.df = single_record
        
        summary = self.analytics.get_daily_summary()
        self.assertEqual(len(summary), 1)
        
        # Test with missing columns
        missing_columns = pd.DataFrame({
            'timestamp': ['2024-01-01'],
            'exercise': ['push_ups']
            # Missing 'count' column
        })
        
        self.analytics.df = missing_columns
        
        # Should handle gracefully
        try:
            summary = self.analytics.get_daily_summary()
            self.assertIsInstance(summary, pd.DataFrame)
        except Exception as e:
            self.fail(f"get_daily_summary raised an exception: {e}")

class TestWorkoutAnalyticsIntegration(unittest.TestCase):
    """Integration tests for WorkoutAnalytics"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        with patch('ai_fitness.core.workout_analytics.get_settings') as mock_settings:
            mock_settings.return_value = Mock(
                workout_log_file='test_integration.csv'
            )
            self.analytics = WorkoutAnalytics()
    
    def test_full_workflow(self):
        """Test complete analytics workflow"""
        # Create comprehensive test data
        test_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='D'),
            'exercise': ['push_ups', 'squats', 'pull_ups', 'lunges', 'planks'] * 20,
            'count': np.random.randint(5, 100, 100),
            'duration': np.random.randint(15, 180, 100),
            'intensity': ['low', 'medium', 'high'] * 33 + ['low']
        })
        
        self.analytics.df = test_data
        
        # Test all analytics methods
        daily_summary = self.analytics.get_daily_summary()
        weekly_progress = self.analytics.get_weekly_progress()
        
        # Verify results
        self.assertGreater(len(daily_summary), 0)
        self.assertGreater(len(weekly_progress), 0)
        
        # Test that data is properly aggregated
        total_exercises = daily_summary['count'].sum()
        self.assertEqual(total_exercises, test_data['count'].sum())
    
    def test_performance_with_large_dataset(self):
        """Test performance with larger dataset"""
        # Create larger dataset
        large_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2020-01-01', periods=1000, freq='D'),
            'exercise': np.random.choice(['push_ups', 'squats', 'pull_ups', 'lunges'], 1000),
            'count': np.random.randint(1, 200, 1000),
            'duration': np.random.randint(10, 300, 1000)
        })
        
        self.analytics.df = large_data
        
        # Time the operations
        import time
        
        start_time = time.time()
        daily_summary = self.analytics.get_daily_summary()
        daily_time = time.time() - start_time
        
        start_time = time.time()
        weekly_progress = self.analytics.get_weekly_progress()
        weekly_time = time.time() - start_time
        
        # Verify performance is reasonable (should complete in under 1 second)
        self.assertLess(daily_time, 1.0, f"Daily summary took {daily_time:.3f} seconds")
        self.assertLess(weekly_time, 1.0, f"Weekly progress took {weekly_time:.3f} seconds")
        
        # Verify results
        self.assertEqual(len(daily_summary), len(large_data.groupby(['timestamp', 'exercise']).size()))

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
