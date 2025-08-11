# AI Fitness Coach 💪

A comprehensive AI-powered fitness coaching application that uses computer vision to track and analyze your workout sessions in real-time.

## Features

- **Real-time Exercise Detection**: Uses MediaPipe and OpenCV to detect and count exercises
- **Multiple Exercise Types**: Supports squats, push-ups, and planks
- **Interactive Web Interface**: Beautiful Streamlit-based dashboard
- **Workout Analytics**: Comprehensive analytics and progress tracking
- **Data Persistence**: Saves workout data for long-term progress tracking

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam for exercise detection
- Windows 10/11 (tested on Windows)

### Installation

1. **Clone or download the project**
   ```bash
   # If you have git
   git clone <repository-url>
   cd AI-fitness
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   ```bash
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

#### Option 1: Using the startup script (Recommended)
```bash
python run_app.py
```

#### Option 2: Direct Streamlit command
```bash
python -m streamlit run Ai_fitness/streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

## 📱 How to Use

### Web Interface (Streamlit App)

1. **Exercise Selection**: Use the sidebar buttons to switch between exercises
2. **Real-time Video Streaming**: Live camera feed with pose detection
3. **Real-time Stats**: View your current session statistics
4. **Analytics**: Explore your workout history and progress with real data
5. **Manual Counting**: Use manual buttons to add reps during testing



## 🏋️ Exercise Detection

### Squats
- **Detection**: Tracks knee angle to count repetitions
- **Form**: Ensures proper depth and form
- **Feedback**: Visual angle display and rep counting

### Push-ups
- **Detection**: Monitors arm angle and body position
- **Form**: Ensures proper elbow bend and body alignment
- **Feedback**: Visual angle display and rep counting

### Planks
- **Detection**: Tracks body alignment and hold duration
- **Form**: Ensures straight body position
- **Feedback**: Visual angle display and rep counting

## 📊 Analytics Features

- **Daily Activity Tracking**: Visualize your daily exercise patterns
- **Exercise Comparison**: Compare performance across different exercises
- **Weekly Trends**: Track progress over time
- **Workout Reports**: Comprehensive session summaries
- **Progress Visualization**: Interactive charts and graphs

## 🛠️ Project Structure

```
AI fitness/
├── Ai_fitness/
│   ├── streamlit_app.py      # Main web application
│   └── workout_analytics.py  # Analytics and data processing
├── scripts/
│   └── test_camera.py       # Camera testing utility
├── logs/                    # Workout data storage
│   └── workout_log.csv     # Exercise logs
├── requirements.txt         # Python dependencies
├── run_app.py              # Startup script
├── test_streamlit.py       # Testing script
├── start_app.bat           # Windows startup script
└── README.md               # This file
```

## 🔧 Troubleshooting

### Common Issues

1. **Camera not working**
   - Ensure your webcam is connected and not being used by another application
   - Check camera permissions in your browser/system settings
   - Position yourself 3-6 feet from the camera with good lighting

2. **Import errors**
   - Make sure you've activated the virtual environment
   - Verify all dependencies are installed: `pip install -r requirements.txt`

3. **Session state errors**
   - Clear your browser cache and refresh the page
   - Restart the Streamlit application

4. **Performance issues**
   - Close other applications using the camera
   - Ensure good lighting for better pose detection

### Testing

Run the test script to verify everything is working:

```bash
python test_streamlit.py
```

## 📈 Data Management

### Workout Data
- Data is automatically saved to `logs/workout_log.csv`
- Data persists between sessions

### Analytics
- View historical data in the analytics section
- Export data for external analysis
- Track long-term progress and trends

## 🤝 Contributing

Feel free to contribute to this project by:
- Reporting bugs
- Suggesting new features
- Improving exercise detection algorithms
- Enhancing the user interface

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **MediaPipe**: For pose detection and tracking
- **OpenCV**: For computer vision capabilities
- **Streamlit**: For the web interface framework
- **Plotly**: For interactive visualizations

---

**Happy Exercising! 💪**

For support or questions, please check the troubleshooting section or create an issue in the repository. 