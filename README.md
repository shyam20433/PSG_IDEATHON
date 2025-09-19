# AI-Powered Video Meeting Platform with Attention Monitoring

A comprehensive Google Meet clone with advanced AI-powered features including real-time attention monitoring, automated Minutes of Meeting (MOM) generation, and user activity tracking.

## 🚀 Features Overview

### 📹 **Core Video Conferencing**
- **Real-time Video & Audio**: WebRTC-based peer-to-peer communication
- **Multi-participant Support**: Up to 50 participants per meeting
- **Modern UI**: Google Meet-inspired responsive design
- **Meeting Management**: Create, join, and manage meetings with unique IDs
- **Chat Integration**: Real-time text messaging during meetings

### 🤖 **AI-Powered Attention Monitoring**
- **Eye Tracking**: ML-based eye detection and attention scoring
- **Face Recognition**: Human presence detection using MediaPipe
- **Engagement Analysis**: Real-time participant engagement levels
- **Activity Status**: Active/Idle/Inactive user monitoring
- **Admin Dashboard**: Creator-only access to attention analytics

### 📝 **Automated Meeting Documentation**
- **Voice Transcription**: Real-time speech-to-text conversion
- **NLP-Powered MOM**: Automated Minutes of Meeting generation
- **Advanced Summarization**: Using BART transformer models
- **Sentiment Analysis**: Meeting tone and participant sentiment
- **Topic Modeling**: Automatic key topic extraction

### 👥 **User Activity Management**
- **Real-time Status Tracking**: Live participant activity monitoring
- **Admin Controls**: Meeting creator privileges
- **Activity Dashboard**: Visual activity overview
- **API Endpoints**: RESTful API for external monitoring
- **Auto-refresh**: Real-time updates every 5 seconds

## 🛠 Technology Stack

### **Backend Framework**
- **Flask**: Web framework for server-side logic
- **Flask-SocketIO**: Real-time bidirectional communication
- **Eventlet**: Async server handling for better performance

### **Real-time Communication**
- **WebRTC**: Peer-to-peer video/audio streaming
- **Socket.IO**: Real-time messaging and signaling
- **STUN Servers**: NAT traversal for WebRTC connections

### **AI & Machine Learning**
- **MediaPipe**: Computer vision for facial detection
- **OpenCV**: Image processing and frame analysis
- **Transformers (BART)**: Advanced text summarization
- **spaCy**: Natural language processing
- **NLTK**: Text processing and analysis
- **Sentence Transformers**: Semantic similarity analysis

### **Frontend Technologies**
- **HTML5**: Modern web standards
- **CSS3**: Responsive design and animations
- **JavaScript ES6+**: Interactive user interface
- **Chart.js**: Real-time data visualization

## 📁 Project Structure

```
meeting/
├── app.py                          # Main Flask application
├── templates/                      # HTML templates
│   ├── index.html                 # Home page
│   ├── join.html                  # Meeting join page
│   ├── meeting.html               # Main meeting interface
│   ├── attention_dashboard.html   # Real-time monitoring dashboard
│   ├── attention_reports.html     # Meeting analytics reports
│   ├── admin_panel.html           # Admin activity monitoring
│   └── error.html                 # Error pages
├── meeting_monitor/               # AI monitoring modules
│   └── attention_monitor.py       # Core ML monitoring logic
├── nlp_models/                    # NLP processing modules
│   └── mom_generator.py           # Minutes of Meeting generator
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── ARCHITECTURE.md                # System architecture diagrams
```

## 🏗️ System Architecture

For detailed system architecture diagrams, data flow charts, and component relationships, see **[ARCHITECTURE.md](ARCHITECTURE.md)**.

Key architectural components:
- **Frontend**: HTML5/CSS3/JavaScript with WebRTC
- **Backend**: Flask + SocketIO + Eventlet
- **AI/ML**: MediaPipe + OpenCV + Transformers
- **Real-time**: WebRTC P2P + Socket.IO signaling

## 🚀 Installation & Setup

### **Prerequisites**
- Python 3.8+
- Webcam and microphone
- Modern web browser (Chrome, Firefox, Safari)

### **Installation Steps**

1. **Clone the project**:
   ```bash
   cd "c:\Users\shyam\OneDrive\Desktop\New folder\meeting"
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Required packages**:
   ```
   Flask==2.3.3
   Flask-SocketIO==5.3.6
   eventlet==0.33.3
   opencv-python==4.8.1.78
   mediapipe==0.10.7
   transformers==4.35.0
   torch==2.1.0
   spacy==3.7.2
   nltk==3.8.1
   sentence-transformers==2.2.2
   numpy==1.24.3
   ```

4. **Download spaCy model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Start the server**:
   ```bash
   python app.py
   ```

6. **Access the application**:
   - Open browser to `http://localhost:8002`

## 📖 Usage Guide

### **Creating a Meeting**
1. Go to the home page
2. Enter meeting title and your name
3. Click "Create Meeting"
4. Share the meeting ID with participants

### **Joining a Meeting**
1. Enter the meeting ID on home page
2. Provide your name
3. Allow camera/microphone permissions
4. Join the meeting

### **Admin Features (Meeting Creator)**
- **Recording Control**: Start/stop meeting recording
- **Attention Monitoring**: Enable AI-powered engagement tracking
- **Activity Dashboard**: View real-time participant status
- **Reports Access**: Download meeting analytics

### **Attention Monitoring**
1. **Enable Monitoring**: Click the eye (👁️) button
2. **View Dashboard**: Click "Monitor" to open real-time dashboard
3. **Check Activity**: Use admin panel (`/admin`) for quick status
4. **API Access**: Use `/api/activity-status/{meeting_id}` endpoint

## 🔧 API Endpoints

### **Web Routes**
- `GET /` - Home page
- `POST /create-meeting` - Create new meeting
- `GET /join/{meeting_id}` - Join meeting page
- `GET /meeting/{meeting_id}` - Main meeting interface
- `GET /attention-dashboard/{meeting_id}` - Monitoring dashboard (creator only)
- `GET /attention-reports/{meeting_id}` - Analytics reports (creator only)
- `GET /admin` - Admin activity panel

### **API Endpoints**
- `GET /api/activity-status/{meeting_id}` - User activity status (JSON)
- `GET /download-mom/{meeting_id}` - Download meeting minutes

### **Socket.IO Events**
- `join_meeting` - User joins meeting
- `webrtc_offer/answer` - WebRTC signaling
- `chat_message` - Text messaging
- `start_attention_monitoring` - Enable monitoring
- `video_frame_analysis` - ML frame processing

## 📊 Features in Detail

### **Real-time Attention Monitoring**
- **Eye Aspect Ratio (EAR)**: Calculates eye openness
- **Head Pose Estimation**: Tracks head orientation
- **Distraction Detection**: Identifies looking away, eyes closed
- **Engagement Scoring**: 0-100% attention score
- **Activity Timeouts**: 10s active, 30s inactive thresholds

### **Minutes of Meeting Generation**
- **Automatic Transcription**: Speech-to-text during recording
- **NLP Processing**: Advanced text analysis
- **Key Points Extraction**: Important discussion highlights
- **Participant Analysis**: Individual contribution tracking
- **Sentiment Analysis**: Meeting tone assessment
- **Export Options**: JSON download format

### **User Activity Tracking**
- **Active Status**: Sending video frames (< 10 seconds)
- **Idle Status**: No activity for 10-30 seconds
- **Inactive Status**: No activity for > 30 seconds
- **Visual Indicators**: Color-coded status (green/yellow/red)
- **Real-time Updates**: 5-second refresh intervals

## 🎯 Key Improvements Made

### **Video Display Fixes**
- Enhanced WebRTC peer connection handling
- Improved media stream management
- Better error handling and fallbacks
- Auto-reconnection for failed connections
- Support for audio-only and avatar modes

### **AI Integration**
- MediaPipe facial recognition
- Real-time frame analysis
- Advanced NLP for meeting summaries
- Attention scoring algorithms
- Activity monitoring system

### **User Experience**
- Modern Google Meet-like interface
- Responsive design for all devices
- Real-time chat and notifications
- Visual status indicators
- Comprehensive error handling

## 🔒 Security & Privacy

- **Creator-only Access**: Sensitive features restricted to meeting creator
- **Session Management**: Secure user session handling
- **Data Privacy**: Temporary file cleanup after processing
- **Secure Communication**: HTTPS-ready architecture

## 🚀 Performance Optimizations

- **Eventlet Server**: Async handling for better performance
- **Frame Rate Control**: Optimized video analysis intervals
- **Memory Management**: Automatic cleanup of resources
- **Connection Pooling**: Efficient WebRTC connection management

## 🐛 Troubleshooting

### **Common Issues**

1. **Camera/Microphone Access Denied**:
   - Allow permissions in browser settings
   - Refresh the page after granting permissions

2. **Video Not Displaying**:
   - Check browser compatibility (Chrome recommended)
   - Ensure WebRTC is enabled
   - Check network firewall settings

3. **Port Conflicts**:
   - Server runs on port 8002 by default
   - Change port in `app.py` if needed

4. **ML Libraries Not Working**:
   - Install OpenCV: `pip install opencv-python`
   - Install MediaPipe: `pip install mediapipe`

## 📈 Future Enhancements

- **Recording & Playback**: Video recording functionality
- **Screen Sharing**: Desktop sharing capabilities
- **Mobile App**: Native mobile applications
- **Cloud Deployment**: AWS/Azure hosting support
- **Advanced Analytics**: Enhanced meeting insights
- **Integration APIs**: Third-party service connections

## 🤝 Contributing

This project was developed as a comprehensive video conferencing solution with AI-powered features. The codebase is modular and extensible for future enhancements.

## 📝 License

Educational/Personal use. Please ensure compliance with all third-party library licenses.

---

**Built with ❤️ using Flask, WebRTC, and AI/ML technologies**

*Last Updated: 2025-01-19*