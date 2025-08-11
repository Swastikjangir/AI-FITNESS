# Camera Performance Optimization Guide

## Overview

The AI Fitness Coach has been optimized to address camera lag and performance issues. The new system provides three performance modes that balance speed and quality based on your needs and hardware capabilities.

## Performance Modes

### 🚀 Fast Mode
- **Resolution**: 480x360
- **Target FPS**: 15
- **Model Complexity**: Lightweight (0)
- **Best For**: Real-time tracking, lower-end devices, mobile applications
- **Use Case**: Live workout tracking, real-time pose detection

### ⚖️ Balanced Mode (Default)
- **Resolution**: 640x480
- **Target FPS**: 20
- **Model Complexity**: Medium (1)
- **Best For**: General use, good balance of speed and quality
- **Use Case**: Standard workout analysis, balanced performance

### 🎯 Quality Mode
- **Resolution**: 1280x720
- **Target FPS**: 30
- **Model Complexity**: Full (2)
- **Best For**: Detailed analysis, high-end devices
- **Use Case**: Precise form analysis, detailed physique assessment

## Key Performance Improvements

### 1. Frame Skipping
- **Fast Mode**: Processes every 3rd frame
- **Balanced Mode**: Processes every 2nd frame
- **Quality Mode**: Processes every frame

### 2. Asynchronous Processing
- Frame capture and processing run in separate threads
- Non-blocking frame queues prevent lag buildup
- ThreadPoolExecutor for parallel pose detection

### 3. MediaPipe Optimization
- Disabled segmentation for faster processing
- Configurable model complexity
- Smooth landmark tracking enabled
- Reduced confidence thresholds for faster detection

### 4. Camera Buffer Optimization
- Minimal buffer size (1 frame)
- MJPG codec for better performance
- Automatic frame resizing for performance modes

## How to Use

### In Streamlit App
1. Go to the "📷 AI Analysis" page
2. Select your desired performance mode using the radio buttons
3. Start the camera
4. Monitor performance stats using the "📊 Performance Stats" button

### In Command Line
```bash
# Test camera performance with different modes
python scripts/test_camera_performance.py

# Run AI analysis with performance mode switching
python run_app.py --ai-analysis
# Press 'f' for fast mode, 'b' for balanced, 'q' for quality
```

### Programmatically
```python
from ai_fitness.services.camera_service import OptimizedCameraService
from ai_fitness.core.ai_analyzer import AIAnalyzer

# Initialize with performance mode
camera = OptimizedCameraService(performance_mode="fast")
ai_analyzer = AIAnalyzer(performance_mode="fast")

# Change mode on the fly
camera.set_performance_mode("balanced")
ai_analyzer.set_performance_mode("balanced")
```

## Performance Monitoring

### Real-time Stats
- Current FPS vs Target FPS
- Frame queue size
- Processing intervals
- Performance mode status

### Performance Tips
1. **Start with Balanced Mode**: Best default for most users
2. **Use Fast Mode for Real-time**: When you need smooth tracking
3. **Use Quality Mode for Analysis**: When accuracy is more important than speed
4. **Good Lighting**: Better lighting improves detection speed
5. **Clear Camera View**: Ensure camera has unobstructed view of subject

## Troubleshooting

### High Lag Issues
1. Switch to "Fast" mode
2. Check system resources (CPU, memory)
3. Ensure good lighting conditions
4. Close other applications using camera

### Poor Detection Quality
1. Switch to "Quality" mode
2. Improve lighting conditions
3. Ensure full body is visible
4. Check camera positioning

### Performance Comparison
Run the performance test script to compare modes:
```bash
python scripts/test_camera_performance.py
```

## Technical Details

### Frame Processing Pipeline
1. **Capture**: Camera captures frame at hardware FPS
2. **Queue**: Frame added to small buffer queue
3. **Skip**: Frame skipping based on performance mode
4. **Process**: Asynchronous pose detection
5. **Display**: Processed frame shown to user

### Threading Architecture
- **Main Thread**: UI and user interaction
- **Camera Thread**: Frame capture and streaming
- **Processing Thread**: Pose detection and analysis
- **Display Thread**: Frame rendering and overlay

### Memory Management
- Small frame buffers prevent memory buildup
- Automatic cleanup of old frames
- Efficient numpy array handling
- Minimal object creation during processing

## Future Improvements

- GPU acceleration for pose detection
- Adaptive performance mode switching
- Machine learning-based frame importance scoring
- Hardware-specific optimizations
- Real-time performance analytics dashboard

## Support

If you experience performance issues:
1. Check the performance mode settings
2. Run the performance test script
3. Monitor system resources
4. Try different performance modes
5. Report issues with performance statistics
