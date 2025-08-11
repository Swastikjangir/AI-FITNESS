"""
AI Analyzer module for AI Fitness Coach.

Provides AI-powered physique analysis, workout generation, and fitness
recommendations. This module is resilient to environments where optional
native dependencies (OpenCV/MediaPipe) are unavailable (e.g., Streamlit Cloud).
"""

# Optional native dependencies
try:
    import cv2  # type: ignore
    OPENCV_AVAILABLE = True
    CV2_IMPORT_ERROR = ""
except Exception as _e:  # pragma: no cover
    OPENCV_AVAILABLE = False
    CV2_IMPORT_ERROR = str(_e)

if OPENCV_AVAILABLE:
    try:
        import mediapipe as mp  # type: ignore
        MEDIAPIPE_AVAILABLE = True
        MP_IMPORT_ERROR = ""
    except Exception as _e:  # pragma: no cover
        MEDIAPIPE_AVAILABLE = False
        MP_IMPORT_ERROR = str(_e)
        mp = None  # type: ignore
else:
    MEDIAPIPE_AVAILABLE = False
    MP_IMPORT_ERROR = "OpenCV not available; skipping MediaPipe import"
    mp = None  # type: ignore

import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime

from ai_fitness.config.settings import get_settings

class PhysiqueAnalyzer:
    """AI-powered physique analysis and fitness recommendations"""
    
    def __init__(self, performance_mode: str = "balanced"):
        self.settings = get_settings()
        self.performance_mode = performance_mode
        self.pose = None
        
        # Initialize only if dependencies available
        if not (OPENCV_AVAILABLE and MEDIAPIPE_AVAILABLE and mp is not None):
            print(
                "PhysiqueAnalyzer running in degraded mode: OpenCV/MediaPipe unavailable. "
                f"cv2_ok={OPENCV_AVAILABLE}, mp_ok={MEDIAPIPE_AVAILABLE}"
            )
            self.mp_pose = None
            return

        # Configure MediaPipe based on performance mode
        if performance_mode == "fast":
            model_complexity = 0  # Lightweight model
            min_detection_confidence = 0.5
            min_tracking_confidence = 0.5
        elif performance_mode == "balanced":
            model_complexity = 1  # Medium model
            min_detection_confidence = 0.6
            min_tracking_confidence = 0.6
        else:  # quality mode
            model_complexity = 2  # Full model
            min_detection_confidence = 0.7
            min_tracking_confidence = 0.7
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            smooth_landmarks=True,
            smooth_segmentation=False
        )
        
        # Body composition estimation parameters
        self.body_types = {
            'ectomorph': {'characteristics': ['lean', 'tall', 'narrow'], 'recommendations': 'muscle_gain'},
            'mesomorph': {'characteristics': ['athletic', 'muscular', 'medium'], 'recommendations': 'strength'},
            'endomorph': {'characteristics': ['round', 'soft', 'wide'], 'recommendations': 'fat_loss'}
        }
        
        print(f"PhysiqueAnalyzer initialized with {performance_mode} mode")
        print(f"Model complexity: {model_complexity}, Detection confidence: {min_detection_confidence}")
        
    def set_performance_mode(self, mode: str):
        """Change performance mode on the fly"""
        if mode in ["fast", "balanced", "quality"]:
            self.performance_mode = mode
            # Reinitialize pose detection with new settings
            self.__init__(mode)
        else:
            print(f"Invalid performance mode: {mode}. Use 'fast', 'balanced', or 'quality'")
    
    def analyze_physique(self, image: np.ndarray) -> Dict:
        """Analyze physique from image and return body composition data"""
        try:
            if not (OPENCV_AVAILABLE and MEDIAPIPE_AVAILABLE and self.pose is not None):
                return {
                    'error': 'AI analysis is unavailable in this environment (missing OpenCV/MediaPipe).',
                    'cv2_available': OPENCV_AVAILABLE,
                    'mediapipe_available': MEDIAPIPE_AVAILABLE,
                    'cv2_error': CV2_IMPORT_ERROR,
                    'mediapipe_error': MP_IMPORT_ERROR,
                }
            # Resize image for performance if needed
            if OPENCV_AVAILABLE:
                if self.performance_mode == "fast" and (image.shape[1] > 480 or image.shape[0] > 360):
                    image = cv2.resize(image, (480, 360), interpolation=cv2.INTER_NEAREST)
                elif self.performance_mode == "balanced" and (image.shape[1] > 640 or image.shape[0] > 480):
                    image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_NEAREST)
            
            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if OPENCV_AVAILABLE else image
            results = self.pose.process(image_rgb)
            
            if not results.pose_landmarks:
                return {'error': 'No pose detected. Please ensure full body is visible.'}
            
            landmarks = results.pose_landmarks.landmark
            
            # Extract key measurements
            measurements = self._extract_measurements(landmarks, image.shape)
            body_type = self._classify_body_type(measurements)
            fitness_level = self._assess_fitness_level(landmarks)
            
            return {
                'body_type': body_type,
                'fitness_level': fitness_level,
                'measurements': measurements,
                'recommendations': self._generate_recommendations(body_type, fitness_level),
                'timestamp': datetime.now().isoformat(),
                'performance_mode': self.performance_mode
            }
            
        except Exception as e:
            return {'error': f'Analysis failed: {str(e)}'}
    
    def _extract_measurements(self, landmarks: List, image_shape: Tuple) -> Dict:
        """Extract body measurements from pose landmarks"""
        height, width = image_shape[:2]
        
        # Get key body points
        shoulders = [
            (landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width,
             landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height),
            (landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * width,
             landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * height)
        ]
        
        hips = [
            (landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * width,
             landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * height),
            (landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x * width,
             landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y * height)
        ]
        
        # Calculate proportions
        shoulder_width = np.linalg.norm(np.array(shoulders[1]) - np.array(shoulders[0]))
        hip_width = np.linalg.norm(np.array(hips[1]) - np.array(hips[0]))
        torso_height = np.mean([shoulders[0][1], shoulders[1][1]]) - np.mean([hips[0][1], hips[1][1]])
        
        # Body proportions
        shoulder_to_hip_ratio = shoulder_width / hip_width if hip_width > 0 else 1.0
        torso_proportion = torso_height / height if height > 0 else 0.3
        
        return {
            'shoulder_width': shoulder_width,
            'hip_width': hip_width,
            'torso_height': torso_height,
            'shoulder_to_hip_ratio': shoulder_to_hip_ratio,
            'torso_proportion': torso_proportion,
            'overall_height': height
        }
    
    def _classify_body_type(self, measurements: Dict) -> str:
        """Classify body type based on measurements"""
        shoulder_hip_ratio = measurements['shoulder_to_hip_ratio']
        torso_prop = measurements['torso_proportion']
        
        # Classification logic based on body proportions
        if shoulder_hip_ratio > 1.1 and torso_prop > 0.4:
            return 'mesomorph'  # Athletic build
        elif shoulder_hip_ratio < 0.9 and torso_prop < 0.35:
            return 'ectomorph'  # Lean build
        else:
            return 'endomorph'  # Rounder build
    
    def _assess_fitness_level(self, landmarks: List) -> str:
        """Assess current fitness level based on pose quality"""
        # Analyze pose stability and alignment
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        # Check shoulder alignment
        shoulder_alignment = abs(left_shoulder.y - right_shoulder.y)
        hip_alignment = abs(left_hip.y - right_hip.y)
        
        # Check posture
        posture_score = (shoulder_alignment + hip_alignment) / 2
        
        if posture_score < 0.05:
            return 'advanced'
        elif posture_score < 0.1:
            return 'intermediate'
        else:
            return 'beginner'
    
    def _generate_recommendations(self, body_type: str, fitness_level: str) -> Dict:
        """Generate personalized workout and diet recommendations"""
        
        # Workout recommendations based on body type and fitness level
        workout_plans = {
            'ectomorph': {
                'beginner': {
                    'focus': 'muscle_gain',
                    'frequency': '3-4 days/week',
                    'intensity': 'moderate',
                    'rest_days': '2-3 days',
                    'exercises': ['compound_movements', 'progressive_overload', 'adequate_rest']
                },
                'intermediate': {
                    'focus': 'muscle_gain',
                    'frequency': '4-5 days/week',
                    'intensity': 'moderate_to_high',
                    'rest_days': '2 days',
                    'exercises': ['strength_training', 'hypertrophy', 'proper_nutrition']
                },
                'advanced': {
                    'focus': 'muscle_gain',
                    'frequency': '5-6 days/week',
                    'intensity': 'high',
                    'rest_days': '1-2 days',
                    'exercises': ['advanced_techniques', 'periodization', 'optimal_nutrition']
                }
            },
            'mesomorph': {
                'beginner': {
                    'focus': 'strength_and_definition',
                    'frequency': '3-4 days/week',
                    'intensity': 'moderate',
                    'rest_days': '2-3 days',
                    'exercises': ['balanced_training', 'form_focus', 'consistency']
                },
                'intermediate': {
                    'focus': 'strength_and_definition',
                    'frequency': '4-5 days/week',
                    'intensity': 'moderate_to_high',
                    'rest_days': '2 days',
                    'exercises': ['strength_training', 'cardio', 'flexibility']
                },
                'advanced': {
                    'focus': 'strength_and_definition',
                    'frequency': '5-6 days/week',
                    'intensity': 'high',
                    'rest_days': '1-2 days',
                    'exercises': ['advanced_strength', 'sports_specific', 'recovery_focus']
                }
            },
            'endomorph': {
                'beginner': {
                    'focus': 'fat_loss',
                    'frequency': '4-5 days/week',
                    'intensity': 'moderate',
                    'rest_days': '2-3 days',
                    'exercises': ['cardio', 'strength_training', 'diet_focus']
                },
                'intermediate': {
                    'focus': 'fat_loss',
                    'frequency': '5-6 days/week',
                    'intensity': 'moderate_to_high',
                    'rest_days': '1-2 days',
                    'exercises': ['hiit', 'strength_training', 'calorie_deficit']
                },
                'advanced': {
                    'focus': 'fat_loss',
                    'frequency': '6 days/week',
                    'intensity': 'high',
                    'rest_days': '1 day',
                    'exercises': ['advanced_cardio', 'strength_training', 'strict_nutrition']
                }
            }
        }
        
        # Diet recommendations
        diet_plans = {
            'ectomorph': {
                'calories': 'surplus_300_500',
                'protein': '1.6-2.2g/kg',
                'carbs': '4-7g/kg',
                'fats': '0.8-1.2g/kg',
                'meals': '5-6 per day',
                'focus': 'muscle_building_nutrition'
            },
            'mesomorph': {
                'calories': 'maintenance_to_slight_surplus',
                'protein': '1.4-2.0g/kg',
                'carbs': '3-5g/kg',
                'fats': '0.8-1.0g/kg',
                'meals': '4-5 per day',
                'focus': 'balanced_nutrition'
            },
            'endomorph': {
                'calories': 'deficit_300_500',
                'protein': '1.6-2.2g/kg',
                'carbs': '2-4g/kg',
                'fats': '0.6-1.0g/kg',
                'meals': '3-4 per day',
                'focus': 'fat_loss_nutrition'
            }
        }
        
        return {
            'workout_plan': workout_plans[body_type][fitness_level],
            'diet_plan': diet_plans[body_type],
            'body_type': body_type,
            'fitness_level': fitness_level
        }

class WorkoutGenerator:
    """Generate specific workout routines based on AI analysis"""
    
    def __init__(self):
        self.exercise_database = self._load_exercise_database()
    
    def _load_exercise_database(self) -> Dict:
        """Load comprehensive exercise database"""
        return {
            'strength_training': {
                'compound_movements': [
                    {'name': 'Squats', 'sets': 3, 'reps': '8-12', 'rest': '90s', 'difficulty': 'intermediate'},
                    {'name': 'Deadlifts', 'sets': 3, 'reps': '6-10', 'rest': '120s', 'difficulty': 'advanced'},
                    {'name': 'Bench Press', 'sets': 3, 'reps': '8-12', 'rest': '90s', 'difficulty': 'intermediate'},
                    {'name': 'Overhead Press', 'sets': 3, 'reps': '8-12', 'rest': '90s', 'difficulty': 'intermediate'},
                    {'name': 'Pull-ups', 'sets': 3, 'reps': '6-12', 'rest': '90s', 'difficulty': 'intermediate'}
                ],
                'isolation_movements': [
                    {'name': 'Bicep Curls', 'sets': 3, 'reps': '10-15', 'rest': '60s', 'difficulty': 'beginner'},
                    {'name': 'Tricep Dips', 'sets': 3, 'reps': '8-12', 'rest': '60s', 'difficulty': 'intermediate'},
                    {'name': 'Lateral Raises', 'sets': 3, 'reps': '12-15', 'rest': '60s', 'difficulty': 'beginner'}
                ]
            },
            'cardio': {
                'low_intensity': [
                    {'name': 'Walking', 'duration': '30-45min', 'intensity': 'low', 'frequency': 'daily'},
                    {'name': 'Cycling', 'duration': '30-45min', 'intensity': 'low', 'frequency': '3-4x/week'}
                ],
                'high_intensity': [
                    {'name': 'HIIT Sprints', 'duration': '20-30min', 'intensity': 'high', 'frequency': '2-3x/week'},
                    {'name': 'Circuit Training', 'duration': '30-45min', 'intensity': 'high', 'frequency': '3x/week'}
                ]
            },
            'flexibility': [
                {'name': 'Dynamic Stretching', 'duration': '10-15min', 'frequency': 'pre_workout'},
                {'name': 'Static Stretching', 'duration': '15-20min', 'frequency': 'post_workout'},
                {'name': 'Yoga', 'duration': '30-45min', 'frequency': '2-3x/week'}
            ]
        }
    
    def generate_workout_routine(self, analysis: Dict) -> Dict:
        """Generate complete workout routine based on AI analysis"""
        recommendations = analysis['recommendations']
        workout_plan = recommendations['workout_plan']
        fitness_level = analysis['fitness_level']
        
        # Generate weekly schedule
        weekly_routine = self._create_weekly_schedule(workout_plan, fitness_level)
        
        # Generate specific exercises for each day
        detailed_routine = {}
        for day, focus in weekly_routine.items():
            detailed_routine[day] = self._generate_day_workout(focus, fitness_level)
        
        return {
            'weekly_schedule': weekly_routine,
            'detailed_routine': detailed_routine,
            'recommendations': recommendations,
            'generated_at': datetime.now().isoformat()
        }
    
    def _create_weekly_schedule(self, workout_plan: Dict, fitness_level: str) -> Dict:
        """Create weekly workout schedule"""
        frequency = workout_plan['frequency']
        focus = workout_plan['focus']
        
        if '3-4' in frequency:
            days = ['Monday', 'Wednesday', 'Friday']
        elif '4-5' in frequency:
            days = ['Monday', 'Tuesday', 'Thursday', 'Friday']
        elif '5-6' in frequency:
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        else:  # 6 days
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        
        schedule = {}
        for i, day in enumerate(days):
            if focus == 'muscle_gain':
                if i % 2 == 0:
                    schedule[day] = 'upper_body'
                else:
                    schedule[day] = 'lower_body'
            elif focus == 'fat_loss':
                schedule[day] = 'full_body_cardio'
            else:  # strength_and_definition
                if i % 3 == 0:
                    schedule[day] = 'push'
                elif i % 3 == 1:
                    schedule[day] = 'pull'
                else:
                    schedule[day] = 'legs'
        
        return schedule
    
    def _generate_day_workout(self, focus: str, fitness_level: str) -> Dict:
        """Generate specific exercises for a given day"""
        exercises = []
        
        if focus == 'upper_body':
            exercises = [
                {'name': 'Bench Press', 'sets': 4, 'reps': '8-12', 'rest': '90s'},
                {'name': 'Overhead Press', 'sets': 3, 'reps': '8-12', 'rest': '90s'},
                {'name': 'Pull-ups', 'sets': 3, 'reps': '6-12', 'rest': '90s'},
                {'name': 'Bicep Curls', 'sets': 3, 'reps': '10-15', 'rest': '60s'},
                {'name': 'Tricep Dips', 'sets': 3, 'reps': '8-12', 'rest': '60s'}
            ]
        elif focus == 'lower_body':
            exercises = [
                {'name': 'Squats', 'sets': 4, 'reps': '8-12', 'rest': '120s'},
                {'name': 'Deadlifts', 'sets': 3, 'reps': '6-10', 'rest': '120s'},
                {'name': 'Lunges', 'sets': 3, 'reps': '10-15 each leg', 'rest': '90s'},
                {'name': 'Calf Raises', 'sets': 3, 'reps': '15-20', 'rest': '60s'}
            ]
        elif focus == 'full_body_cardio':
            exercises = [
                {'name': 'Circuit Training', 'sets': 4, 'duration': '5min', 'rest': '2min'},
                {'name': 'HIIT Sprints', 'sets': 8, 'duration': '30s', 'rest': '90s'},
                {'name': 'Bodyweight Squats', 'sets': 3, 'reps': '15-20', 'rest': '60s'},
                {'name': 'Push-ups', 'sets': 3, 'reps': '10-15', 'rest': '60s'}
            ]
        elif focus == 'push':
            exercises = [
                {'name': 'Bench Press', 'sets': 4, 'reps': '8-12', 'rest': '90s'},
                {'name': 'Overhead Press', 'sets': 3, 'reps': '8-12', 'rest': '90s'},
                {'name': 'Incline Dumbbell Press', 'sets': 3, 'reps': '10-12', 'rest': '90s'},
                {'name': 'Tricep Dips', 'sets': 3, 'reps': '8-12', 'rest': '60s'}
            ]
        elif focus == 'pull':
            exercises = [
                {'name': 'Pull-ups', 'sets': 4, 'reps': '6-12', 'rest': '90s'},
                {'name': 'Barbell Rows', 'sets': 3, 'reps': '8-12', 'rest': '90s'},
                {'name': 'Lat Pulldowns', 'sets': 3, 'reps': '10-12', 'rest': '90s'},
                {'name': 'Bicep Curls', 'sets': 3, 'reps': '10-15', 'rest': '60s'}
            ]
        elif focus == 'legs':
            exercises = [
                {'name': 'Squats', 'sets': 4, 'reps': '8-12', 'rest': '120s'},
                {'name': 'Deadlifts', 'sets': 3, 'reps': '6-10', 'rest': '120s'},
                {'name': 'Leg Press', 'sets': 3, 'reps': '10-15', 'rest': '90s'},
                {'name': 'Calf Raises', 'sets': 3, 'reps': '15-20', 'rest': '60s'}
            ]
        
        return {
            'focus': focus,
            'exercises': exercises,
            'total_duration': '45-60 minutes',
            'warm_up': '5-10 minutes dynamic stretching',
            'cool_down': '5-10 minutes static stretching'
        }

class AIAnalyzer:
    """Main AI Analyzer class that combines physique analysis and workout generation"""
    
    def __init__(self, performance_mode: str = "balanced"):
        self.performance_mode = performance_mode
        self.physique_analyzer = PhysiqueAnalyzer(performance_mode)
        self.workout_generator = WorkoutGenerator()
        
        print(f"AIAnalyzer initialized with {performance_mode} performance mode")
    
    def set_performance_mode(self, mode: str):
        """Change performance mode for both physique analyzer and camera service"""
        if mode in ["fast", "balanced", "quality"]:
            self.performance_mode = mode
            self.physique_analyzer.set_performance_mode(mode)
            print(f"AIAnalyzer performance mode changed to: {mode}")
        else:
            print(f"Invalid performance mode: {mode}. Use 'fast', 'balanced', or 'quality'")
    
    def get_performance_mode(self) -> str:
        """Get current performance mode"""
        return self.performance_mode
    
    def analyzeAndGenerate(self, image: np.ndarray) -> Dict:
        """Analyze physique and generate workout routine in one call"""
        # Analyze physique
        analysis = self.physique_analyzer.analyze_physique(image)
        
        if 'error' in analysis:
            return analysis
        
        # Generate workout routine
        workout_routine = self.workout_generator.generate_workout_routine(analysis)
        
        return {
            'analysis': analysis,
            'workout_routine': workout_routine,
            'performance_mode': self.performance_mode
        }
    
    def analyze_physique_only(self, image: np.ndarray) -> Dict:
        """Analyze physique only"""
        return self.physique_analyzer.analyze_physique(image)
    
    def generate_workout_only(self, analysis: Dict) -> Dict:
        """Generate workout routine only"""
        return self.workout_generator.generate_workout_routine(analysis)
