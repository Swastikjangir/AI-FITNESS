"""
Unit tests for AI Analyzer module.

Tests the PhysiqueAnalyzer, WorkoutGenerator, and AIAnalyzer classes.
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_fitness.core.ai_analyzer import PhysiqueAnalyzer, WorkoutGenerator, AIAnalyzer

class TestPhysiqueAnalyzer(unittest.TestCase):
    """Test cases for PhysiqueAnalyzer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = PhysiqueAnalyzer()
        
        # Mock image data
        self.mock_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Mock pose landmarks
        self.mock_landmarks = []
        for i in range(33):  # MediaPipe pose has 33 landmarks
            landmark = Mock()
            landmark.x = np.random.random()
            landmark.y = np.random.random()
            landmark.z = np.random.random()
            self.mock_landmarks.append(landmark)
    
    def test_init(self):
        """Test PhysiqueAnalyzer initialization"""
        self.assertIsNotNone(self.analyzer.mp_pose)
        self.assertIsNotNone(self.analyzer.pose)
        self.assertIn('ectomorph', self.analyzer.body_types)
        self.assertIn('mesomorph', self.analyzer.body_types)
        self.assertIn('endomorph', self.analyzer.body_types)
    
    @patch('ai_fitness.core.ai_analyzer.mp.solutions.pose.Pose.process')
    def test_analyze_physique_success(self, mock_process):
        """Test successful physique analysis"""
        # Mock successful pose detection
        mock_results = Mock()
        mock_results.pose_landmarks.landmark = self.mock_landmarks
        mock_process.return_value = mock_results
        
        result = self.analyzer.analyze_physique(self.mock_image)
        
        self.assertNotIn('error', result)
        self.assertIn('body_type', result)
        self.assertIn('fitness_level', result)
        self.assertIn('measurements', result)
        self.assertIn('recommendations', result)
        self.assertIn('timestamp', result)
    
    @patch('ai_fitness.core.ai_analyzer.mp.solutions.pose.Pose.process')
    def test_analyze_physique_no_pose(self, mock_process):
        """Test physique analysis when no pose is detected"""
        # Mock no pose detection
        mock_results = Mock()
        mock_results.pose_landmarks = None
        mock_process.return_value = mock_results
        
        result = self.analyzer.analyze_physique(self.mock_image)
        
        self.assertIn('error', result)
        self.assertIn('No pose detected', result['error'])
    
    def test_extract_measurements(self):
        """Test body measurements extraction"""
        measurements = self.analyzer._extract_measurements(self.mock_landmarks, (480, 640, 3))
        
        self.assertIn('shoulder_width', measurements)
        self.assertIn('hip_width', measurements)
        self.assertIn('torso_height', measurements)
        self.assertIn('shoulder_to_hip_ratio', measurements)
        self.assertIn('torso_proportion', measurements)
        self.assertIn('overall_height', measurements)
    
    def test_classify_body_type(self):
        """Test body type classification"""
        # Test mesomorph (athletic build)
        measurements = {
            'shoulder_to_hip_ratio': 1.2,
            'torso_proportion': 0.45
        }
        body_type = self.analyzer._classify_body_type(measurements)
        self.assertEqual(body_type, 'mesomorph')
        
        # Test ectomorph (lean build)
        measurements = {
            'shoulder_to_hip_ratio': 0.8,
            'torso_proportion': 0.3
        }
        body_type = self.analyzer._classify_body_type(measurements)
        self.assertEqual(body_type, 'ectomorph')
        
        # Test endomorph (rounder build)
        measurements = {
            'shoulder_to_hip_ratio': 1.0,
            'torso_proportion': 0.38
        }
        body_type = self.analyzer._classify_body_type(measurements)
        self.assertEqual(body_type, 'endomorph')
    
    def test_assess_fitness_level(self):
        """Test fitness level assessment"""
        # Test advanced fitness level
        landmarks = self.mock_landmarks
        # Set landmarks to simulate good posture
        landmarks[11].y = 0.5  # Left shoulder
        landmarks[12].y = 0.5  # Right shoulder
        landmarks[23].y = 0.6  # Left hip
        landmarks[24].y = 0.6  # Right hip
        
        fitness_level = self.analyzer._assess_fitness_level(landmarks)
        self.assertEqual(fitness_level, 'advanced')
    
    def test_generate_recommendations(self):
        """Test recommendation generation"""
        recommendations = self.analyzer._generate_recommendations('mesomorph', 'intermediate')
        
        self.assertIn('workout_plan', recommendations)
        self.assertIn('diet_plan', recommendations)
        self.assertEqual(recommendations['body_type'], 'mesomorph')
        self.assertEqual(recommendations['fitness_level'], 'intermediate')
        
        workout_plan = recommendations['workout_plan']
        self.assertEqual(workout_plan['focus'], 'strength_and_definition')
        self.assertEqual(workout_plan['frequency'], '4-5 days/week')

class TestWorkoutGenerator(unittest.TestCase):
    """Test cases for WorkoutGenerator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.generator = WorkoutGenerator()
        
        # Mock analysis data
        self.mock_analysis = {
            'fitness_level': 'intermediate',
            'recommendations': {
                'workout_plan': {
                    'focus': 'muscle_gain',
                    'frequency': '4-5 days/week'
                }
            }
        }
    
    def test_init(self):
        """Test WorkoutGenerator initialization"""
        self.assertIsNotNone(self.generator.exercise_database)
        self.assertIn('strength_training', self.generator.exercise_database)
        self.assertIn('cardio', self.generator.exercise_database)
        self.assertIn('flexibility', self.generator.exercise_database)
    
    def test_load_exercise_database(self):
        """Test exercise database loading"""
        database = self.generator._load_exercise_database()
        
        # Check strength training exercises
        strength = database['strength_training']
        self.assertIn('compound_movements', strength)
        self.assertIn('isolation_movements', strength)
        
        # Check cardio exercises
        cardio = database['cardio']
        self.assertIn('low_intensity', cardio)
        self.assertIn('high_intensity', cardio)
        
        # Check flexibility exercises
        self.assertIsInstance(database['flexibility'], list)
    
    def test_generate_workout_routine(self):
        """Test workout routine generation"""
        routine = self.generator.generate_workout_routine(self.mock_analysis)
        
        self.assertIn('weekly_schedule', routine)
        self.assertIn('detailed_routine', routine)
        self.assertIn('recommendations', routine)
        self.assertIn('generated_at', routine)
    
    def test_create_weekly_schedule(self):
        """Test weekly schedule creation"""
        workout_plan = {
            'frequency': '4-5 days/week',
            'focus': 'muscle_gain'
        }
        
        schedule = self.generator._create_weekly_schedule(workout_plan, 'intermediate')
        
        self.assertIsInstance(schedule, dict)
        self.assertGreater(len(schedule), 0)
        
        # Check that schedule alternates between upper and lower body
        values = list(schedule.values())
        self.assertIn('upper_body', values)
        self.assertIn('lower_body', values)
    
    def test_generate_day_workout(self):
        """Test daily workout generation"""
        # Test upper body workout
        workout = self.generator._generate_day_workout('upper_body', 'intermediate')
        
        self.assertEqual(workout['focus'], 'upper_body')
        self.assertIn('exercises', workout)
        self.assertIn('total_duration', workout)
        self.assertIn('warm_up', workout)
        self.assertIn('cool_down', workout)
        
        # Check exercises
        exercises = workout['exercises']
        self.assertGreater(len(exercises), 0)
        
        for exercise in exercises:
            self.assertIn('name', exercise)
            self.assertIn('sets', exercise)
            self.assertIn('reps', exercise)
            self.assertIn('rest', exercise)

class TestAIAnalyzer(unittest.TestCase):
    """Test cases for AIAnalyzer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.ai_analyzer = AIAnalyzer()
        self.mock_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_init(self):
        """Test AIAnalyzer initialization"""
        self.assertIsNotNone(self.ai_analyzer.physique_analyzer)
        self.assertIsNotNone(self.ai_analyzer.workout_generator)
    
    @patch.object(PhysiqueAnalyzer, 'analyze_physique')
    @patch.object(WorkoutGenerator, 'generate_workout_routine')
    def test_analyze_and_generate_success(self, mock_generate, mock_analyze):
        """Test successful analyze and generate workflow"""
        # Mock successful analysis
        mock_analysis = {
            'body_type': 'mesomorph',
            'fitness_level': 'intermediate'
        }
        mock_analyze.return_value = mock_analysis
        
        # Mock successful workout generation
        mock_workout = {
            'weekly_schedule': {'Monday': 'upper_body'},
            'detailed_routine': {}
        }
        mock_generate.return_value = mock_workout
        
        result = self.ai_analyzer.analyzeAndGenerate(self.mock_image)
        
        self.assertIn('analysis', result)
        self.assertIn('workout_routine', result)
        self.assertEqual(result['analysis'], mock_analysis)
        self.assertEqual(result['workout_routine'], mock_workout)
    
    @patch.object(PhysiqueAnalyzer, 'analyze_physique')
    def test_analyze_and_generate_analysis_error(self, mock_analyze):
        """Test analyze and generate when analysis fails"""
        # Mock analysis error
        mock_analyze.return_value = {'error': 'Analysis failed'}
        
        result = self.ai_analyzer.analyzeAndGenerate(self.mock_image)
        
        self.assertIn('error', result)
        self.assertEqual(result['error'], 'Analysis failed')
    
    def test_analyze_physique_only(self):
        """Test physique analysis only"""
        with patch.object(self.ai_analyzer.physique_analyzer, 'analyze_physique') as mock_analyze:
            mock_analyze.return_value = {'body_type': 'mesomorph'}
            
            result = self.ai_analyzer.analyze_physique_only(self.mock_image)
            
            self.assertEqual(result['body_type'], 'mesomorph')
    
    def test_generate_workout_only(self):
        """Test workout generation only"""
        mock_analysis = {'fitness_level': 'beginner'}
        
        with patch.object(self.ai_analyzer.workout_generator, 'generate_workout_routine') as mock_generate:
            mock_generate.return_value = {'schedule': '3 days/week'}
            
            result = self.ai_analyzer.generate_workout_only(mock_analysis)
            
            self.assertEqual(result['schedule'], '3 days/week')

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
