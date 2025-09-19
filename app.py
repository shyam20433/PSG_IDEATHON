from flask import Flask, render_template, request, redirect, url_for, session
from flask_socketio import SocketIO, emit, join_room, leave_room, close_room, rooms, disconnect
import uuid
import time
from datetime import datetime
import json
import os
import tempfile
import speech_recognition as sr
from pydub import AudioSegment
import threading
import time
from collections import defaultdict

# Computer Vision Libraries
try:
    import cv2
    import mediapipe as mp
    from meeting_monitor import MeetingMonitor
    CV_ENABLED = True
    print("Computer Vision libraries loaded successfully!")
except ImportError as e:
    print(f"Computer Vision libraries not found: {e}")
    print("Install with: pip install opencv-python mediapipe")
    CV_ENABLED = False

# NLP Libraries
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    import nltk
    import spacy
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    import numpy as np
    import pandas as pd
    
    # Download required NLTK data
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    
    # Load models
    print("Loading NLP models...")
    BART_MODEL = "facebook/bart-large-cnn"
    summarizer = pipeline("summarization", model=BART_MODEL)
    sentiment_analyzer = pipeline("sentiment-analysis")
    
    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Please install spaCy English model: python -m spacy download en_core_web_sm")
        nlp = None
    
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    NLP_ENABLED = True
    print("NLP models loaded successfully!")
except ImportError as e:
    print(f"NLP libraries not found: {e}")
    print("Install with: pip install -r requirements.txt")
    NLP_ENABLED = False

app = Flask(__name__)
app.config['SECRET_KEY'] = 'video_meeting_secret_key_2025'

# Initialize SocketIO with CORS enabled
socketio = SocketIO(app, cors_allowed_origins="*", engineio_logger=True, socketio_logger=True)

# In-memory storage for meetings and participants
active_meetings = {}
meeting_participants = {}
meeting_transcripts = {}  # Store transcripts for each meeting
meeting_recordings = {}   # Store audio recordings for each meeting
meeting_monitors = {}     # Store attention monitors for each meeting

class MOMGenerator:
    """Advanced Minutes of Meeting Generator using NLP"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        
    def extract_key_points(self, text):
        """Extract key points using TF-IDF and clustering"""
        if not NLP_ENABLED or not text.strip():
            return ["No key points available"]
        
        try:
            sentences = nltk.sent_tokenize(text)
            if len(sentences) < 2:
                return sentences
            
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Cluster sentences
            n_clusters = min(3, len(sentences))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(tfidf_matrix)
            
            # Get representative sentences from each cluster
            key_points = []
            for i in range(n_clusters):
                cluster_sentences = [sentences[j] for j in range(len(sentences)) if clusters[j] == i]
                if cluster_sentences:
                    # Get the sentence closest to cluster center
                    cluster_indices = [j for j in range(len(sentences)) if clusters[j] == i]
                    center = kmeans.cluster_centers_[i]
                    distances = [np.linalg.norm(tfidf_matrix[j].toarray() - center) for j in cluster_indices]
                    closest_idx = cluster_indices[np.argmin(distances)]
                    key_points.append(sentences[closest_idx])
            
            return key_points[:5]  # Return top 5 key points
        except Exception as e:
            print(f"Error extracting key points: {e}")
            return ["Error extracting key points"]
    
    def extract_action_items(self, text):
        """Extract action items using keyword matching and NER"""
        if not text.strip():
            return []
        
        action_keywords = [
            'need to', 'should', 'must', 'will', 'action item', 'todo', 'follow up',
            'assign', 'responsible', 'deadline', 'complete', 'finish', 'deliver'
        ]
        
        sentences = nltk.sent_tokenize(text)
        action_items = []
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in action_keywords):
                action_items.append(sentence.strip())
        
        return action_items[:10]  # Return top 10 action items
    
    def extract_decisions(self, text):
        """Extract decisions made during the meeting"""
        if not text.strip():
            return []
        
        decision_keywords = [
            'decided', 'agreed', 'resolved', 'concluded', 'determined',
            'final decision', 'we will', 'consensus', 'vote', 'approved'
        ]
        
        sentences = nltk.sent_tokenize(text)
        decisions = []
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in decision_keywords):
                decisions.append(sentence.strip())
        
        return decisions[:10]  # Return top 10 decisions
    
    def extract_participants(self, text):
        """Extract participant names using NER"""
        if not NLP_ENABLED or not nlp or not text.strip():
            return []
        
        try:
            doc = nlp(text)
            participants = set()
            
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    participants.add(ent.text)
            
            return list(participants)[:20]  # Return up to 20 participants
        except Exception as e:
            print(f"Error extracting participants: {e}")
            return []
    
    def analyze_sentiment(self, text):
        """Analyze overall sentiment of the meeting"""
        if not NLP_ENABLED or not text.strip():
            return {"label": "NEUTRAL", "score": 0.5}
        
        try:
            result = sentiment_analyzer(text[:512])  # Limit text length
            return result[0]
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return {"label": "NEUTRAL", "score": 0.5}
    
    def extract_topics(self, text):
        """Extract main topics discussed"""
        if not NLP_ENABLED or not text.strip():
            return ["General Discussion"]
        
        try:
            # Use sentence embeddings to find topic clusters
            sentences = nltk.sent_tokenize(text)
            if len(sentences) < 2:
                return ["General Discussion"]
            
            embeddings = sentence_model.encode(sentences)
            
            # Cluster sentences to find topics
            n_clusters = min(5, len(sentences))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(embeddings)
            
            topics = []
            for i in range(n_clusters):
                cluster_sentences = [sentences[j] for j in range(len(sentences)) if clusters[j] == i]
                if cluster_sentences:
                    # Use first sentence as topic representative
                    topic = cluster_sentences[0][:50] + "..." if len(cluster_sentences[0]) > 50 else cluster_sentences[0]
                    topics.append(topic)
            
            return topics[:5]  # Return top 5 topics
        except Exception as e:
            print(f"Error extracting topics: {e}")
            return ["General Discussion"]
    
    def generate_summary(self, text):
        """Generate abstractive summary using BART"""
        if not NLP_ENABLED or not text.strip():
            return "No content available for summary."
        
        try:
            # Split text into chunks if too long
            max_chunk_length = 1024
            chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
            
            summaries = []
            for chunk in chunks[:3]:  # Limit to first 3 chunks
                if len(chunk.strip()) > 50:  # Only summarize substantial chunks
                    summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
                    summaries.append(summary[0]['summary_text'])
            
            return " ".join(summaries) if summaries else "Meeting content too brief for summary."
        except Exception as e:
            print(f"Error generating summary: {e}")
            return "Error generating summary."
    
    def generate_mom(self, meeting_id, transcript_text, meeting_info):
        """Generate complete Minutes of Meeting"""
        if not transcript_text.strip():
            transcript_text = "No audio content was captured during this meeting."
        
        # Extract various components
        summary = self.generate_summary(transcript_text)
        key_points = self.extract_key_points(transcript_text)
        action_items = self.extract_action_items(transcript_text)
        decisions = self.extract_decisions(transcript_text)
        participants = self.extract_participants(transcript_text)
        sentiment = self.analyze_sentiment(transcript_text)
        topics = self.extract_topics(transcript_text)
        
        # Create MOM structure
        mom = {
            "meeting_id": meeting_id,
            "meeting_title": meeting_info.get('title', 'Meeting'),
            "date": datetime.now().strftime('%Y-%m-%d'),
            "time": meeting_info.get('start_time', datetime.now().strftime('%H:%M')),
            "duration": meeting_info.get('duration', 'N/A'),
            "organizer": meeting_info.get('creator', 'Unknown'),
            "participants_detected": participants,
            "participants_joined": meeting_info.get('participants', []),
            "summary": summary,
            "key_points": key_points,
            "action_items": action_items,
            "decisions": decisions,
            "topics_discussed": topics,
            "sentiment_analysis": {
                "overall_sentiment": sentiment.get('label', 'NEUTRAL'),
                "confidence": round(sentiment.get('score', 0.5), 2)
            },
            "full_transcript": transcript_text,
            "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return mom

class MeetingRoom:
    def __init__(self, room_id, creator_name, meeting_title="New Meeting"):
        self.room_id = room_id
        self.creator_name = creator_name
        self.meeting_title = meeting_title
        self.created_at = datetime.now()
        self.participants = {}
        self.is_active = True
        self.max_participants = 50
        self.is_recording = False
        self.recording_start_time = None
        self.transcript_text = ""
        self.mom_generated = False
        self.attention_monitoring = False
        self.attention_reports = {}
    
    def add_participant(self, user_id, user_name, socket_id):
        if len(self.participants) >= self.max_participants:
            return False
        
        self.participants[user_id] = {
            'name': user_name,
            'socket_id': socket_id,
            'joined_at': datetime.now(),
            'is_muted': False,
            'is_video_off': False,
            'is_creator': user_id == self.creator_name
        }
        return True
    
    def remove_participant(self, user_id):
        if user_id in self.participants:
            del self.participants[user_id]
        
        # Close meeting if creator leaves or no participants
        if not self.participants or user_id == self.creator_name:
            self.is_active = False
            # Generate MOM if recording was active
            if self.is_recording and not self.mom_generated:
                self.generate_mom()
    
    def start_recording(self):
        """Start recording the meeting"""
        if not self.is_recording:
            self.is_recording = True
            self.recording_start_time = datetime.now()
            self.transcript_text = ""
            
            # Start attention monitoring if CV is enabled
            if CV_ENABLED and not self.attention_monitoring:
                self.start_attention_monitoring()
            
            print(f"Recording started for meeting {self.room_id}")
    
    def stop_recording(self):
        """Stop recording and generate MOM"""
        if self.is_recording:
            self.is_recording = False
            
            # Stop attention monitoring
            if self.attention_monitoring:
                self.stop_attention_monitoring()
            
            print(f"Recording stopped for meeting {self.room_id}")
            if not self.mom_generated:
                self.generate_mom()
    
    def add_transcript(self, text):
        """Add text to the meeting transcript"""
        if self.is_recording:
            self.transcript_text += f" {text}"
    
    def generate_mom(self):
        """Generate Minutes of Meeting"""
        if self.mom_generated:
            return
        
        try:
            mom_generator = MOMGenerator()
            meeting_info = {
                'title': self.meeting_title,
                'creator': self.creator_name,
                'start_time': self.recording_start_time.strftime('%H:%M') if self.recording_start_time else 'N/A',
                'duration': self.get_meeting_duration(),
                'participants': [p['name'] for p in self.participants.values()]
            }
            
            mom = mom_generator.generate_mom(self.room_id, self.transcript_text, meeting_info)
            
            # Store MOM
            meeting_transcripts[self.room_id] = mom
            self.mom_generated = True
            
            print(f"MOM generated for meeting {self.room_id}")
            return mom
        except Exception as e:
            print(f"Error generating MOM: {e}")
            return None
    
    def get_meeting_duration(self):
        """Get meeting duration"""
        if self.recording_start_time:
            duration = datetime.now() - self.recording_start_time
            hours, remainder = divmod(duration.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        return "N/A"
    
    def start_attention_monitoring(self):
        """Start attention monitoring for the meeting"""
        if CV_ENABLED and not self.attention_monitoring:
            self.attention_monitoring = True
            monitor = MeetingMonitor()
            monitor.start_monitoring(self.room_id, self.creator_name)
            meeting_monitors[self.room_id] = monitor
            
            # Add all current participants to monitoring
            for user_id in self.participants.keys():
                monitor.add_participant(user_id)
            
            print(f"Attention monitoring started for meeting {self.room_id}")
    
    def stop_attention_monitoring(self):
        """Stop attention monitoring and get final reports"""
        if self.attention_monitoring and self.room_id in meeting_monitors:
            monitor = meeting_monitors[self.room_id]
            self.attention_reports = monitor.stop_monitoring()
            del meeting_monitors[self.room_id]
            self.attention_monitoring = False
            print(f"Attention monitoring stopped for meeting {self.room_id}")
    
    def add_participant_to_monitoring(self, user_id):
        """Add participant to attention monitoring"""
        if self.attention_monitoring and self.room_id in meeting_monitors:
            meeting_monitors[self.room_id].add_participant(user_id)
    
    def remove_participant_from_monitoring(self, user_id):
        """Remove participant from attention monitoring"""
        if self.attention_monitoring and self.room_id in meeting_monitors:
            report = meeting_monitors[self.room_id].remove_participant(user_id)
            if report:
                self.attention_reports[user_id] = report
    
    def get_participant_list(self):
        return [
            {
                'id': user_id,
                'name': info['name'],
                'is_muted': info['is_muted'],
                'is_video_off': info['is_video_off'],
                'is_creator': info['is_creator'],
                'joined_at': info['joined_at'].strftime('%H:%M')
            }
            for user_id, info in self.participants.items()
        ]

@app.route('/')
def index():
    """Home page with meeting creation options"""
    return render_template('index.html')

@app.route('/create-meeting', methods=['POST'])
def create_meeting():
    """Create a new meeting room"""
    meeting_title = request.form.get('meeting_title', 'New Meeting')
    creator_name = request.form.get('creator_name', 'Anonymous')
    
    # Generate unique meeting ID
    meeting_id = str(uuid.uuid4())[:8].upper()
    
    # Create meeting room
    meeting_room = MeetingRoom(meeting_id, creator_name, meeting_title)
    active_meetings[meeting_id] = meeting_room
    
    # Store creator info in session
    session['user_name'] = creator_name
    session['user_id'] = creator_name
    
    return redirect(f'/join/{meeting_id}')

@app.route('/join/<meeting_id>')
def join_meeting_page(meeting_id):
    """Page to join an existing meeting"""
    if meeting_id not in active_meetings:
        return render_template('error.html', 
                             message="Meeting not found or has ended",
                             error_code="404")
    
    meeting = active_meetings[meeting_id]
    if not meeting.is_active:
        return render_template('error.html',
                             message="This meeting has ended",
                             error_code="ENDED")
    
    return render_template('join.html', meeting_id=meeting_id, meeting=meeting)

@app.route('/meeting/<meeting_id>')
def meeting_room(meeting_id):
    """Main meeting room interface"""
    if meeting_id not in active_meetings:
        return redirect(url_for('index'))
    
    meeting = active_meetings[meeting_id]
    if not meeting.is_active:
        return redirect(url_for('index'))
    
    user_name = session.get('user_name', 'Anonymous')
    user_id = session.get('user_id', str(uuid.uuid4()))
    
    return render_template('meeting.html', 
                         meeting_id=meeting_id,
                         meeting=meeting,
                         user_name=user_name,
                         user_id=user_id)

@app.route('/join-meeting', methods=['POST'])
def join_meeting():
    """Join an existing meeting"""
    meeting_id = request.form.get('meeting_id', '').upper()
    user_name = request.form.get('user_name', 'Anonymous')
    
    if not meeting_id or meeting_id not in active_meetings:
        return render_template('join.html', 
                             error="Meeting ID not found",
                             meeting_id=meeting_id)
    
    meeting = active_meetings[meeting_id]
    if not meeting.is_active:
        return render_template('join.html',
                             error="This meeting has ended",
                             meeting_id=meeting_id)
    
    if len(meeting.participants) >= meeting.max_participants:
        return render_template('join.html',
                             error="Meeting is full",
                             meeting_id=meeting_id)
    
    # Store user info in session
    session['user_name'] = user_name
    session['user_id'] = user_name + "_" + str(uuid.uuid4())[:4]
    
    return redirect(url_for('meeting_room', meeting_id=meeting_id))

@app.route('/mom/<meeting_id>')
def view_mom(meeting_id):
    """View Minutes of Meeting for a specific meeting"""
    if meeting_id in meeting_transcripts:
        mom = meeting_transcripts[meeting_id]
        return render_template('mom.html', mom=mom)
    else:
        return render_template('error.html', 
                             message="Minutes of Meeting not found or not yet generated",
                             error_code="404")

@app.route('/download-mom/<meeting_id>')
def download_mom(meeting_id):
    """Download MOM as JSON file"""
    if meeting_id in meeting_transcripts:
        mom = meeting_transcripts[meeting_id]
        
        from flask import jsonify, make_response
        response = make_response(jsonify(mom))
        response.headers['Content-Disposition'] = f'attachment; filename=MOM_{meeting_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        response.headers['Content-Type'] = 'application/json'
        return response
    else:
        return render_template('error.html', 
                             message="Minutes of Meeting not found",
                             error_code="404")

@app.route('/attention-dashboard/<meeting_id>')
def attention_dashboard(meeting_id):
    """Attention monitoring dashboard (creator only)"""
    if meeting_id not in active_meetings:
        return render_template('error.html', 
                             message="Meeting not found",
                             error_code="404")
    
    meeting = active_meetings[meeting_id]
    user_id = session.get('user_id', '')
    
    # Check if user is the creator
    if user_id != meeting.creator_name:
        return render_template('error.html',
                             message="Access denied. Only meeting creators can view attention monitoring.",
                             error_code="403")
    
    # Get real-time stats if monitoring is active
    stats = None
    if meeting_id in meeting_monitors:
        stats = meeting_monitors[meeting_id].get_real_time_stats()
    
    return render_template('attention_dashboard.html', 
                         meeting_id=meeting_id,
                         meeting=meeting,
                         stats=stats,
                         cv_enabled=CV_ENABLED)

@app.route('/attention-reports/<meeting_id>')
def attention_reports(meeting_id):
    """Final attention reports for a meeting"""
    if meeting_id not in active_meetings:
        return render_template('error.html', 
                             message="Meeting not found",
                             error_code="404")
    
    meeting = active_meetings[meeting_id]
    user_id = session.get('user_id', '')
    
    # Check if user is the creator
    if user_id != meeting.creator_name:
        return render_template('error.html',
                             message="Access denied. Only meeting creators can view attention reports.",
                             error_code="403")
    
    return render_template('attention_reports.html',
                         meeting_id=meeting_id,
                         meeting=meeting,
                         reports=meeting.attention_reports)

@app.route('/admin')
def admin_panel():
    """Admin panel for monitoring user activity across meetings"""
    return render_template('admin_panel.html')

@app.route('/api/activity-status/<meeting_id>')
def get_activity_status(meeting_id):
    """API endpoint to get quick activity status of all participants"""
    if meeting_id not in active_meetings:
        return {'error': 'Meeting not found'}, 404
    
    meeting = active_meetings[meeting_id]
    user_id = session.get('user_id', '')
    
    # Check if user is the creator
    if user_id != meeting.creator_name:
        return {'error': 'Access denied. Only meeting creators can view activity status.'}, 403
    
    if meeting_id in meeting_monitors:
        activity_summary = meeting_monitors[meeting_id].get_activity_summary()
        return {
            'meeting_id': meeting_id,
            'total_participants': len(meeting.participants),
            'activity_summary': activity_summary,
            'timestamp': time.time()
        }
    else:
        return {
            'meeting_id': meeting_id,
            'total_participants': len(meeting.participants),
            'activity_summary': {
                'active_users': [],
                'idle_users': [],
                'inactive_users': [],
                'total_active': 0,
                'total_idle': 0,
                'total_inactive': 0
            },
            'monitoring_active': False,
            'timestamp': time.time()
        }

# SocketIO Event Handlers
@socketio.on('connect')
def on_connect():
    print(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def on_disconnect():
    print(f"Client disconnected: {request.sid}")
    
    # Remove user from all meetings (create a copy to avoid runtime error)
    for meeting_id, meeting in list(active_meetings.items()):
        for user_id, participant in list(meeting.participants.items()):
            if participant['socket_id'] == request.sid:
                # Remove from attention monitoring
                meeting.remove_participant_from_monitoring(user_id)
                
                meeting.remove_participant(user_id)
                
                # Notify other participants
                emit('participant_left', {
                    'user_id': user_id,
                    'participants': meeting.get_participant_list()
                }, room=meeting_id, include_self=False)
                
                # Close meeting if creator left
                if not meeting.is_active:
                    emit('meeting_ended', {
                        'message': 'Meeting ended by host'
                    }, room=meeting_id)
                    close_room(meeting_id)
                    if meeting_id in active_meetings:
                        del active_meetings[meeting_id]

@socketio.on('join_meeting')
def on_join_meeting(data):
    """Handle user joining a meeting"""
    meeting_id = data['meeting_id']
    user_name = data['user_name']
    user_id = data['user_id']
    
    if meeting_id not in active_meetings:
        emit('error', {'message': 'Meeting not found'})
        return
    
    meeting = active_meetings[meeting_id]
    
    if not meeting.add_participant(user_id, user_name, request.sid):
        emit('error', {'message': 'Cannot join meeting - room is full'})
        return
    
    # Join the room
    join_room(meeting_id)
    
    # Add to attention monitoring if active
    if meeting.attention_monitoring:
        meeting.add_participant_to_monitoring(user_id)
    
    # Notify user of successful join
    emit('joined_meeting', {
        'meeting_id': meeting_id,
        'meeting_title': meeting.meeting_title,
        'user_id': user_id,
        'participants': meeting.get_participant_list()
    })
    
    # Notify other participants
    emit('participant_joined', {
        'user_id': user_id,
        'user_name': user_name,
        'participants': meeting.get_participant_list()
    }, room=meeting_id, include_self=False)

@socketio.on('leave_meeting')
def on_leave_meeting(data):
    """Handle user leaving a meeting"""
    meeting_id = data['meeting_id']
    user_id = data['user_id']
    
    if meeting_id in active_meetings:
        meeting = active_meetings[meeting_id]
        
        # Remove from attention monitoring
        meeting.remove_participant_from_monitoring(user_id)
        
        meeting.remove_participant(user_id)
        
        leave_room(meeting_id)
        
        # Notify other participants
        emit('participant_left', {
            'user_id': user_id,
            'participants': meeting.get_participant_list()
        }, room=meeting_id)
        
        # Close meeting if creator left or no participants
        if not meeting.is_active:
            emit('meeting_ended', {
                'message': 'Meeting ended by host'
            }, room=meeting_id)
            close_room(meeting_id)
            del active_meetings[meeting_id]

@socketio.on('toggle_mute')
def on_toggle_mute(data):
    """Handle mute/unmute toggle"""
    meeting_id = data['meeting_id']
    user_id = data['user_id']
    is_muted = data['is_muted']
    
    if meeting_id in active_meetings:
        meeting = active_meetings[meeting_id]
        if user_id in meeting.participants:
            meeting.participants[user_id]['is_muted'] = is_muted
            
            # Notify other participants
            emit('participant_mute_changed', {
                'user_id': user_id,
                'is_muted': is_muted,
                'participants': meeting.get_participant_list()
            }, room=meeting_id, include_self=False)

@socketio.on('toggle_video')
def on_toggle_video(data):
    """Handle video on/off toggle"""
    meeting_id = data['meeting_id']
    user_id = data['user_id']
    is_video_off = data['is_video_off']
    
    if meeting_id in active_meetings:
        meeting = active_meetings[meeting_id]
        if user_id in meeting.participants:
            meeting.participants[user_id]['is_video_off'] = is_video_off
            
            # Notify other participants
            emit('participant_video_changed', {
                'user_id': user_id,
                'is_video_off': is_video_off,
                'participants': meeting.get_participant_list()
            }, room=meeting_id, include_self=False)

@socketio.on('remove_participant')
def on_remove_participant(data):
    """Handle removing a participant (host only)"""
    meeting_id = data['meeting_id']
    user_id = data['user_id']
    target_user_id = data['target_user_id']
    
    if meeting_id in active_meetings:
        meeting = active_meetings[meeting_id]
        
        # Check if user is the creator
        if user_id in meeting.participants and meeting.participants[user_id]['is_creator']:
            # Remove target participant
            if target_user_id in meeting.participants:
                target_socket_id = meeting.participants[target_user_id]['socket_id']
                meeting.remove_participant(target_user_id)
                
                # Notify removed participant
                emit('removed_from_meeting', {
                    'message': 'You have been removed from the meeting'
                }, room=target_socket_id)
                
                # Notify other participants
                emit('participant_left', {
                    'user_id': target_user_id,
                    'participants': meeting.get_participant_list()
                }, room=meeting_id)

# WebRTC Signaling Events
@socketio.on('webrtc_offer')
def on_webrtc_offer(data):
    """Handle WebRTC offer"""
    target_user_id = data['target_user_id']
    meeting_id = data['meeting_id']
    
    if meeting_id in active_meetings:
        meeting = active_meetings[meeting_id]
        if target_user_id in meeting.participants:
            target_socket_id = meeting.participants[target_user_id]['socket_id']
            emit('webrtc_offer', {
                'offer': data['offer'],
                'sender_user_id': data['sender_user_id']
            }, room=target_socket_id)

@socketio.on('webrtc_answer')
def on_webrtc_answer(data):
    """Handle WebRTC answer"""
    target_user_id = data['target_user_id']
    meeting_id = data['meeting_id']
    
    if meeting_id in active_meetings:
        meeting = active_meetings[meeting_id]
        if target_user_id in meeting.participants:
            target_socket_id = meeting.participants[target_user_id]['socket_id']
            emit('webrtc_answer', {
                'answer': data['answer'],
                'sender_user_id': data['sender_user_id']
            }, room=target_socket_id)

@socketio.on('webrtc_ice_candidate')
def on_webrtc_ice_candidate(data):
    """Handle WebRTC ICE candidates"""
    target_user_id = data['target_user_id']
    meeting_id = data['meeting_id']
    
    if meeting_id in active_meetings:
        meeting = active_meetings[meeting_id]
        if target_user_id in meeting.participants:
            target_socket_id = meeting.participants[target_user_id]['socket_id']
            emit('webrtc_ice_candidate', {
                'candidate': data['candidate'],
                'sender_user_id': data['sender_user_id']
            }, room=target_socket_id)

@socketio.on('chat_message')
def on_chat_message(data):
    """Handle chat messages"""
    meeting_id = data['meeting_id']
    user_id = data['user_id']
    message = data['message']
    
    if meeting_id in active_meetings:
        meeting = active_meetings[meeting_id]
        if user_id in meeting.participants:
            user_name = meeting.participants[user_id]['name']
            
            # Add message to transcript if recording
            if meeting.is_recording:
                meeting.add_transcript(f"{user_name} (chat): {message}")
            
            # Broadcast message to all participants
            emit('chat_message', {
                'user_id': user_id,
                'user_name': user_name,
                'message': message,
                'timestamp': datetime.now().strftime('%H:%M')
            }, room=meeting_id)

@socketio.on('start_recording')
def on_start_recording(data):
    """Start recording the meeting"""
    meeting_id = data['meeting_id']
    user_id = data['user_id']
    
    if meeting_id in active_meetings:
        meeting = active_meetings[meeting_id]
        
        # Only creator can start recording
        if user_id in meeting.participants and meeting.participants[user_id]['is_creator']:
            meeting.start_recording()
            
            # Notify all participants
            emit('recording_started', {
                'message': 'Meeting recording has started',
                'started_by': meeting.participants[user_id]['name']
            }, room=meeting_id)
        else:
            emit('error', {'message': 'Only the meeting creator can start recording'})

@socketio.on('stop_recording')
def on_stop_recording(data):
    """Stop recording and generate MOM"""
    meeting_id = data['meeting_id']
    user_id = data['user_id']
    
    if meeting_id in active_meetings:
        meeting = active_meetings[meeting_id]
        
        # Only creator can stop recording
        if user_id in meeting.participants and meeting.participants[user_id]['is_creator']:
            meeting.stop_recording()
            
            # Generate MOM
            mom = meeting.generate_mom()
            
            # Notify all participants
            emit('recording_stopped', {
                'message': 'Meeting recording has stopped',
                'mom_available': mom is not None,
                'mom_url': f'/mom/{meeting_id}' if mom else None
            }, room=meeting_id)
        else:
            emit('error', {'message': 'Only the meeting creator can stop recording'})

@socketio.on('voice_data')
def on_voice_data(data):
    """Handle voice data for transcription"""
    meeting_id = data['meeting_id']
    user_id = data['user_id']
    
    if meeting_id in active_meetings:
        meeting = active_meetings[meeting_id]
        
        if meeting.is_recording and user_id in meeting.participants:
            try:
                # Process audio data (this would be implemented based on your audio format)
                # For now, we'll simulate transcription
                user_name = meeting.participants[user_id]['name']
                
                # In a real implementation, you would:
                # 1. Convert audio data to proper format
                # 2. Use speech recognition to get text
                # 3. Add to transcript
                
                # Simulated transcription for demo
                if 'text' in data:  # If text is provided directly
                    transcribed_text = data['text']
                    meeting.add_transcript(f"{user_name}: {transcribed_text}")
                    
                    # Broadcast transcription to all participants
                    emit('transcription_update', {
                        'user_name': user_name,
                        'text': transcribed_text,
                        'timestamp': datetime.now().strftime('%H:%M:%S')
                    }, room=meeting_id)
                    
            except Exception as e:
                print(f"Error processing voice data: {e}")
                emit('error', {'message': 'Error processing voice data'})

@socketio.on('start_attention_monitoring')
def on_start_attention_monitoring(data):
    """Start attention monitoring for the meeting"""
    meeting_id = data['meeting_id']
    user_id = data['user_id']
    
    if meeting_id in active_meetings:
        meeting = active_meetings[meeting_id]
        
        # Only creator can start attention monitoring
        if user_id in meeting.participants and meeting.participants[user_id]['is_creator']:
            if CV_ENABLED:
                meeting.start_attention_monitoring()
                
                emit('attention_monitoring_started', {
                    'message': 'Attention monitoring has started',
                    'dashboard_url': f'/attention-dashboard/{meeting_id}'
                }, room=meeting_id)
            else:
                emit('error', {'message': 'Computer Vision libraries not available'})
        else:
            emit('error', {'message': 'Only the meeting creator can start attention monitoring'})

@socketio.on('stop_attention_monitoring')
def on_stop_attention_monitoring(data):
    """Stop attention monitoring"""
    meeting_id = data['meeting_id']
    user_id = data['user_id']
    
    if meeting_id in active_meetings:
        meeting = active_meetings[meeting_id]
        
        # Only creator can stop attention monitoring
        if user_id in meeting.participants and meeting.participants[user_id]['is_creator']:
            meeting.stop_attention_monitoring()
            
            emit('attention_monitoring_stopped', {
                'message': 'Attention monitoring has stopped',
                'reports_url': f'/attention-reports/{meeting_id}'
            }, room=meeting_id)
        else:
            emit('error', {'message': 'Only the meeting creator can stop attention monitoring'})

@socketio.on('video_frame_analysis')
def on_video_frame_analysis(data):
    """Analyze video frame for attention monitoring"""
    meeting_id = data['meeting_id']
    user_id = data['user_id']
    frame_data = data['frame_data']
    
    if meeting_id in meeting_monitors:
        monitor = meeting_monitors[meeting_id]
        
        # Analyze the frame
        analysis = monitor.analyze_participant_frame(user_id, frame_data)
        
        if analysis:
            # Send analysis to meeting creator only
            meeting = active_meetings.get(meeting_id)
            if meeting:
                creator_socket = None
                for participant_id, participant in meeting.participants.items():
                    if participant['is_creator']:
                        creator_socket = participant['socket_id']
                        break
                
                if creator_socket:
                    emit('attention_analysis_update', {
                        'participant_id': user_id,
                        'analysis': analysis
                    }, room=creator_socket)

@socketio.on('get_attention_stats')
def on_get_attention_stats(data):
    """Get real-time attention statistics"""
    meeting_id = data['meeting_id']
    user_id = data['user_id']
    
    if meeting_id in active_meetings:
        meeting = active_meetings[meeting_id]
        
        # Only creator can view stats
        if user_id in meeting.participants and meeting.participants[user_id]['is_creator']:
            if meeting_id in meeting_monitors:
                stats = meeting_monitors[meeting_id].get_real_time_stats()
                emit('attention_stats_update', stats)
            else:
                emit('error', {'message': 'Attention monitoring not active'})
        else:
            emit('error', {'message': 'Access denied'})

if __name__ == '__main__':
    print("Starting Video Meeting Server...")
    print("Open your browser and go to: http://localhost:8002")
    socketio.run(app, debug=False, host='0.0.0.0', port=8002)