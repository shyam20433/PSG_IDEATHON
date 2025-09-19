# Project Architecture & Flow Diagrams

## 🏗️ System Architecture Diagram

```mermaid
graph TB
    subgraph "Frontend Layer"
        A[Web Browser]
        B[HTML5/CSS3/JS]
        C[WebRTC Client]
        D[Socket.IO Client]
    end
    
    subgraph "Backend Layer"
        E[Flask Application]
        F[Flask-SocketIO]
        G[Eventlet Server]
    end
    
    subgraph "AI/ML Processing Layer"
        H[MediaPipe Engine]
        I[OpenCV Processing]
        J[Attention Monitor]
        K[Face Detection]
        L[Eye Tracking]
    end
    
    subgraph "NLP Processing Layer"
        M[BART Summarizer]
        N[spaCy NLP]
        O[NLTK Processor]
        P[Sentence Transformers]
        Q[MOM Generator]
    end
    
    subgraph "Data Storage"
        R[Session Memory]
        S[Meeting Transcripts]
        T[Attention Reports]
        U[User Activity Data]
    end
    
    subgraph "External Services"
        V[STUN Servers]
        W[WebRTC Signaling]
    end
    
    A --> B
    B --> C
    B --> D
    C --> V
    D --> F
    E --> F
    F --> G
    
    E --> H
    H --> I
    I --> J
    J --> K
    J --> L
    
    E --> M
    M --> N
    N --> O
    O --> P
    P --> Q
    
    E --> R
    E --> S
    E --> T
    E --> U
    
    C --> W
    W --> V
```

## 🔄 Application Flow Diagram

```mermaid
flowchart TD
    Start([User Opens Application]) --> Home[Home Page]
    
    Home --> Choice{Create or Join?}
    
    Choice -->|Create| Create[Create Meeting]
    Choice -->|Join| Join[Join Meeting]
    
    Create --> MeetingID[Generate Meeting ID]
    Join --> EnterID[Enter Meeting ID]
    
    MeetingID --> Permissions[Request Camera/Mic]
    EnterID --> Permissions
    
    Permissions --> MediaAccess{Media Access Granted?}
    
    MediaAccess -->|Yes| VideoMode[Full Video Mode]
    MediaAccess -->|Audio Only| AudioMode[Audio Only Mode]
    MediaAccess -->|Denied| AvatarMode[Avatar Mode]
    
    VideoMode --> MeetingRoom[Enter Meeting Room]
    AudioMode --> MeetingRoom
    AvatarMode --> MeetingRoom
    
    MeetingRoom --> Features{Available Features}
    
    Features --> Video[Video/Audio Chat]
    Features --> Chat[Text Chat]
    Features --> Recording[Meeting Recording]
    Features --> Monitoring[Attention Monitoring]
    
    Recording -->|Creator Only| StartRec[Start Recording]
    Monitoring -->|Creator Only| StartMon[Start Monitoring]
    
    StartRec --> Transcription[Real-time Transcription]
    StartMon --> AIAnalysis[AI Frame Analysis]
    
    Transcription --> NLPProcess[NLP Processing]
    AIAnalysis --> AttentionScore[Attention Scoring]
    
    NLPProcess --> MOMGen[Generate MOM]
    AttentionScore --> Dashboard[Real-time Dashboard]
    
    MOMGen --> Export[Export Results]
    Dashboard --> Reports[Activity Reports]
    
    Export --> End([Meeting Ends])
    Reports --> End
    Video --> End
    Chat --> End
```

## 🧠 AI Processing Pipeline

```mermaid
flowchart LR
    subgraph "Video Stream Processing"
        A[Video Frame] --> B[MediaPipe Detection]
        B --> C[Face Landmarks]
        B --> D[Pose Estimation]
        B --> E[Hand Detection]
        
        C --> F[Eye Aspect Ratio]
        D --> G[Head Pose Angles]
        E --> H[Gesture Recognition]
        
        F --> I[Attention Score]
        G --> I
        H --> I
    end
    
    subgraph "Audio Stream Processing"
        J[Audio Stream] --> K[Speech Recognition]
        K --> L[Text Transcription]
        L --> M[Real-time Display]
        L --> N[NLP Pipeline]
    end
    
    subgraph "NLP Analysis"
        N --> O[Text Preprocessing]
        O --> P[Named Entity Recognition]
        O --> Q[Sentiment Analysis]
        O --> R[Topic Modeling]
        
        P --> S[Key Information]
        Q --> S
        R --> S
        
        S --> T[BART Summarization]
        T --> U[MOM Generation]
    end
    
    subgraph "Real-time Updates"
        I --> V[Activity Status]
        M --> W[Live Transcription]
        V --> X[Admin Dashboard]
        W --> X
        U --> Y[Meeting Reports]
    end
```

## 🔌 WebRTC Connection Flow

```mermaid
sequenceDiagram
    participant U1 as User 1
    participant S as Signaling Server
    participant U2 as User 2
    participant STUN as STUN Server
    
    U1->>S: Join Meeting
    S->>U1: Meeting Joined
    
    U2->>S: Join Meeting
    S->>U1: New Participant Joined
    S->>U2: Meeting Joined
    
    U1->>STUN: Get ICE Candidates
    STUN->>U1: ICE Candidates
    
    U1->>S: WebRTC Offer
    S->>U2: WebRTC Offer
    
    U2->>STUN: Get ICE Candidates
    STUN->>U2: ICE Candidates
    
    U2->>S: WebRTC Answer
    S->>U1: WebRTC Answer
    
    U1->>S: ICE Candidates
    S->>U2: ICE Candidates
    
    U2->>S: ICE Candidates
    S->>U1: ICE Candidates
    
    U1-->>U2: Direct P2P Connection Established
    
    Note over U1,U2: Video/Audio Streaming
```

## 📊 Data Flow Architecture

```mermaid
graph LR
    subgraph "Input Sources"
        A[Video Stream]
        B[Audio Stream]
        C[User Actions]
        D[Chat Messages]
    end
    
    subgraph "Processing Engines"
        E[WebRTC Engine]
        F[AI/ML Engine]
        G[NLP Engine]
        H[Session Manager]
    end
    
    subgraph "Storage Layer"
        I[Real-time Memory]
        J[Meeting Data]
        K[User Profiles]
        L[Analytics Data]
    end
    
    subgraph "Output Interfaces"
        M[Live Video Feed]
        N[Real-time Chat]
        O[Attention Dashboard]
        P[Meeting Reports]
        Q[MOM Documents]
    end
    
    A --> E
    A --> F
    B --> E
    B --> G
    C --> H
    D --> H
    
    E --> I
    F --> L
    G --> J
    H --> K
    
    I --> M
    I --> N
    L --> O
    J --> P
    J --> Q
```

## 🏛️ Component Architecture

```mermaid
graph TB
    subgraph "Presentation Layer"
        A[Meeting Interface]
        B[Admin Dashboard]
        C[Activity Monitor]
        D[Reports Viewer]
    end
    
    subgraph "Application Layer"
        E[Meeting Controller]
        F[User Manager]
        G[Recording Controller]
        H[Monitoring Controller]
    end
    
    subgraph "Service Layer"
        I[WebRTC Service]
        J[AI Analysis Service]
        K[NLP Service]
        L[Notification Service]
    end
    
    subgraph "Infrastructure Layer"
        M[Socket.IO Handler]
        N[Session Storage]
        O[File Management]
        P[Error Handler]
    end
    
    A --> E
    B --> F
    C --> H
    D --> F
    
    E --> I
    F --> N
    G --> K
    H --> J
    
    I --> M
    J --> O
    K --> O
    L --> M
    
    M --> P
    N --> P
    O --> P
```

## 🔄 Meeting Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Created: Create Meeting
    Created --> Waiting: Waiting for Participants
    Waiting --> Active: First Participant Joins
    Active --> Recording: Start Recording
    Recording --> Monitoring: Enable AI Monitoring
    Monitoring --> Recording: Continue Recording
    Recording --> Active: Stop Recording
    Monitoring --> Active: Stop Monitoring
    Active --> Waiting: All Participants Leave
    Active --> Ended: Creator Ends Meeting
    Recording --> Ended: Creator Ends Meeting
    Monitoring --> Ended: Creator Ends Meeting
    Waiting --> Ended: Creator Ends Meeting
    Ended --> [*]: Generate Final Reports
```

## 🎯 Feature Integration Map

```mermaid
mindmap
  root((AI Meeting Platform))
    Video Conferencing
      WebRTC P2P
      Multi-participant
      Screen Sharing Ready
      Mobile Responsive
    
    AI Monitoring
      Eye Tracking
      Face Detection
      Attention Scoring
      Activity Status
    
    NLP Processing
      Speech Recognition
      Text Summarization
      Sentiment Analysis
      Topic Modeling
    
    Admin Features
      Meeting Control
      User Management
      Analytics Dashboard
      Report Generation
    
    Real-time Features
      Live Chat
      Video Streaming
      Activity Updates
      Transcription Display
```

## 📈 Performance Optimization Flow

```mermaid
flowchart TD
    A[User Request] --> B{Request Type}
    
    B -->|Video Stream| C[WebRTC Processing]
    B -->|AI Analysis| D[ML Pipeline]
    B -->|Chat Message| E[Socket.IO Handler]
    B -->|File Request| F[Static File Serving]
    
    C --> G[Peer Connection Pool]
    D --> H[Frame Buffer Queue]
    E --> I[Message Broadcasting]
    F --> J[File Cache]
    
    G --> K[Connection Optimization]
    H --> L[Batch Processing]
    I --> M[Real-time Updates]
    J --> N[Static Content Delivery]
    
    K --> O[User Experience]
    L --> O
    M --> O
    N --> O
```