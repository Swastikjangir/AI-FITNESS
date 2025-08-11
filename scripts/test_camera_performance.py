#!/usr/bin/env python3
"""
Camera Performance Test Script

This script tests the optimized camera service with different performance modes
to demonstrate the improvements in camera lag and processing speed.
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_fitness.services.camera_service import OptimizedCameraService
from ai_fitness.core.ai_analyzer import PhysiqueAnalyzer

def test_camera_performance():
    """Test camera performance with different modes"""
    print("🚀 Camera Performance Test")
    print("=" * 50)
    
    # Test different performance modes
    modes = ["fast", "balanced", "quality"]
    
    for mode in modes:
        print(f"\n📹 Testing {mode.upper()} mode...")
        print("-" * 30)
        
        try:
            # Initialize camera service
            camera = OptimizedCameraService(performance_mode=mode)
            
            if not camera.initialize_camera():
                print(f"❌ Failed to initialize camera in {mode} mode")
                continue
            
            # Get camera info
            info = camera.get_camera_info()
            print(f"📊 Camera Info: {info['width']}x{info['height']} @ {info['fps']:.1f} fps")
            print(f"⚡ Performance Mode: {info['performance_mode']}")
            print(f"🔄 Frame Skip: {info['frame_skip']}")
            
            # Test frame capture performance
            print("\n🔄 Testing frame capture performance...")
            frame_count = 0
            start_time = time.time()
            
            # Capture frames for 3 seconds
            while time.time() - start_time < 3.0:
                frame = camera.capture_frame()
                if frame is not None:
                    frame_count += 1
                    # Simulate some processing
                    time.sleep(0.01)
            
            elapsed_time = time.time() - start_time
            actual_fps = frame_count / elapsed_time
            
            print(f"📈 Captured {frame_count} frames in {elapsed_time:.2f} seconds")
            print(f"🎯 Actual FPS: {actual_fps:.1f}")
            print(f"🎯 Target FPS: {info['fps']}")
            print(f"📊 Performance: {(actual_fps/info['fps'])*100:.1f}% of target")
            
            # Test pose detection performance
            print("\n🤖 Testing pose detection performance...")
            pose_analyzer = PhysiqueAnalyzer(performance_mode=mode)
            
            # Get a test frame
            test_frame = camera.capture_frame()
            if test_frame is not None:
                pose_start = time.time()
                result = pose_analyzer.analyze_physique(test_frame)
                pose_time = time.time() - pose_start
                
                if 'error' not in result:
                    print(f"✅ Pose detection successful in {pose_time:.3f} seconds")
                    print(f"📊 Body Type: {result.get('body_type', 'Unknown')}")
                    print(f"💪 Fitness Level: {result.get('fitness_level', 'Unknown')}")
                else:
                    print(f"⚠️ Pose detection: {result['error']}")
            
            # Cleanup
            camera.release_camera()
            
            print(f"✅ {mode.upper()} mode test completed")
            
        except Exception as e:
            print(f"❌ Error testing {mode} mode: {str(e)}")
    
    print("\n🎉 Performance test completed!")
    print("\n💡 Performance Tips:")
    print("- Use 'fast' mode for real-time tracking and lower-end devices")
    print("- Use 'balanced' mode for general use and good performance")
    print("- Use 'quality' mode for detailed analysis and high-end devices")
    print("- Adjust lighting conditions for better pose detection")
    print("- Ensure camera is not blocked and has good view of the subject")

def test_performance_modes_comparison():
    """Compare performance between different modes"""
    print("\n🔍 Performance Mode Comparison")
    print("=" * 50)
    
    comparison_data = {}
    
    for mode in ["fast", "balanced", "quality"]:
        print(f"\n📊 Testing {mode} mode...")
        
        try:
            camera = OptimizedCameraService(performance_mode=mode)
            
            if not camera.initialize_camera():
                continue
            
            # Performance metrics
            start_time = time.time()
            frame_count = 0
            
            # Test for 2 seconds
            while time.time() - start_time < 2.0:
                frame = camera.capture_frame()
                if frame is not None:
                    frame_count += 1
                time.sleep(0.01)
            
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            
            comparison_data[mode] = {
                'fps': fps,
                'resolution': f"{camera.width}x{camera.height}",
                'frame_skip': camera.frame_skip,
                'processing_interval': camera.min_processing_interval
            }
            
            camera.release_camera()
            
        except Exception as e:
            print(f"Error in {mode} mode: {e}")
    
    # Display comparison
    if comparison_data:
        print("\n📊 Performance Comparison Results:")
        print("-" * 60)
        print(f"{'Mode':<10} {'FPS':<8} {'Resolution':<15} {'Frame Skip':<12} {'Processing Interval':<20}")
        print("-" * 60)
        
        for mode, data in comparison_data.items():
            print(f"{mode:<10} {data['fps']:<8.1f} {data['resolution']:<15} {data['frame_skip']:<12} {data['processing_interval']:<20.3f}")
        
        print("\n💡 Recommendations:")
        if 'fast' in comparison_data and 'quality' in comparison_data:
            fast_fps = comparison_data['fast']['fps']
            quality_fps = comparison_data['quality']['fps']
            improvement = ((fast_fps - quality_fps) / quality_fps) * 100
            print(f"- Fast mode provides {improvement:.1f}% better FPS than quality mode")
            print(f"- Use fast mode for real-time applications")
            print(f"- Use quality mode when accuracy is more important than speed")

if __name__ == "__main__":
    try:
        test_camera_performance()
        test_performance_modes_comparison()
        
        print("\n🎯 To run the full AI analysis:")
        print("python run_app.py --ai-analysis")
        print("\n🎯 To run the Streamlit app:")
        print("python run_app.py --streamlit")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        print("Make sure all dependencies are installed and camera is available")
