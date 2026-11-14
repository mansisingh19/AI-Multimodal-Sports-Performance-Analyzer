# AI-Based Multimodal Sports Performance Assessment System

An AI-powered sports performance analysis system that evaluates an athlete's technique from video using **Computer Vision, Deep Learning, Machine Learning, and multimodal feature fusion**.

The system currently supports **Cricket and Badminton** analysis. It extracts body keypoints from sports videos, recognises the performed action/shot, calculates biomechanical and motion-based features, generates a performance score, and provides technique-oriented feedback through an interactive Streamlit dashboard.

---

##  Key Features

-  **Video-based Sports Analysis**
-  **Human Pose Estimation** using MediaPipe
-  **Optional YOLO-based Tracking**
-  **Cricket Shot Recognition**
-  **Badminton Action Recognition**
-  **GRU-based Action Recognition Model**
-  **Biomechanical Feature Extraction**
  - Joint angles
  - Balance
  - Rotation
  - Movement-related features
-  **Motion Analysis** using frame-level motion information
-  **Multimodal Feature Fusion**
-  **Machine Learning-based Performance Scoring**
-  **Rule-based Coaching Feedback**
-  **Evaluation Metrics**
-  **Interactive Streamlit Dashboard**
-  **SQLite-based Analysis History**

---

##  System Architecture

```text
                    Sports Video
                         │
                         ▼
                  Video Processing
                         │
                         ▼
              Pose Estimation / Tracking
                         │
                         ▼
                33 Body Keypoints
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
   Pose Features    Motion Features   Action Model
   / Biomechanics   Frame Differences      │
          │              │             Shot Type
          │              │             + Confidence
          └──────────────┼──────────────┘
                         ▼
                  Feature Fusion
                         │
                         ▼
            Machine Learning Scorer
                         │
                         ▼
               Performance Score
                    /100
                         │
                         ▼
                Feedback Engine
                         │
                         ▼
                Streamlit Dashboard
```

---

##  Multimodal Approach

The system combines multiple information sources extracted from the same sports video:

| Modality | Extracted Information |
|---|---|
| **Pose** | Body/joint keypoints and movement geometry |
| **Biomechanics** | Joint angles, balance, rotation and related features |
| **Video Motion** | Frame-level motion information |
| **Action Recognition** | Shot/action type and model confidence |
| **Deep Features** | Representation learned by the GRU action model |

These features are combined through a **feature-fusion pipeline** before performance scoring.

---

##  Tech Stack

### Programming
- Python

### Computer Vision
- OpenCV
- MediaPipe
- YOLOv8

### Deep Learning
- PyTorch
- GRU-based action recognition

### Machine Learning
- Scikit-learn
- Random Forest
- Gradient Boosting
- XGBoost

### Application
- Streamlit

### Data & Storage
- NumPy
- Pandas
- SQLite

### Development
- VS Code
- Google Colab

---

##  Project Structure

```text
AI-Multimodal-Sports-Performance-Analyzer/
│
├── app/
│   ├── label_tool.py
│   └── streamlit_app.py
│
├── data/
│   ├── keypoints/
│   ├── labels/
│   └── raw_videos/
│
├── models/
│
├── scripts/
│   ├── extract_dataset.py
│   ├── train_action.py
│   ├── train_scoring.py
│   ├── make_label_template.py
│   └── analyze_video.py
│
├── src/
│   ├── config.py
│   ├── pose.py
│   ├── features.py
│   ├── action_model.py
│   ├── scoring.py
│   ├── feedback.py
│   ├── pipeline.py
│   └── db.py
│
├── tests/
│   └── smoke_test.py
│
├── yolov8n.pt
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/mansisingh19/AI-Multimodal-Sports-Performance-Analyzer.git
cd AI-Multimodal-Sports-Performance-Analyzer
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

##  Running the Project

Run the smoke test first:

```bash
python tests/smoke_test.py
```

Then launch the Streamlit dashboard:

```bash
streamlit run app/streamlit_app.py
```

---

##  Cricket Workflow

The project follows the following pipeline for cricket:

```text
Sports Video
     ↓
Pose / Keypoint Extraction
     ↓
Feature Extraction
     ↓
Action Recognition
     ↓
Feature Fusion
     ↓
Performance Scoring
     ↓
Technique Feedback
     ↓
Streamlit Dashboard
```

Example cricket actions can include:

- Cover Drive
- Pull Shot
- Defensive Shot

The exact actions depend on the configured dataset.

---

##  Badminton Workflow

The same pipeline can be used for badminton:

```text
Badminton Video
      ↓
Pose Extraction
      ↓
Feature Extraction
      ↓
Action Recognition
      ↓
Performance Scoring
      ↓
Feedback
```

The sport-specific actions and scoring parameters are configured within the project.

---

##  Model Evaluation

The project includes evaluation of the action-recognition and scoring components.

Action recognition can be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The scoring pipeline can compare:

- Random Forest
- Gradient Boosting
- XGBoost

using metrics such as:

- MAE
- RMSE
- R²

---

##  Dataset

The project is designed to work with sports video datasets organised by:

```text
data/
└── raw_videos/
    ├── cricket/
    │   ├── cover_drive/
    │   ├── pull_shot/
    │   └── defensive_shot/
    │
    └── badminton/
        └── ...
```

For meaningful performance scoring, videos need corresponding human-assigned score labels.

The repository does **not** include the complete raw-video dataset.

---

##  Limitations

- Pose estimation accuracy can decrease with blur, occlusion, or low-resolution videos.
- Joint-angle measurements depend on camera viewpoint.
- The quality of the performance scorer depends on the quality and consistency of human-provided labels.
- Small datasets can lead to overfitting.
- Sports performance scores are subjective and represent the characteristics of the training labels rather than an absolute measure of athletic ability.
- The current multimodal system combines multiple feature types derived from the same video rather than independent sensor modalities.

---

##  Future Scope

Potential improvements include:

- Real-time sports performance analysis
- Support for additional sports
- Multi-camera analysis
- Wearable sensor integration
- Temporal transformer-based action recognition
- More advanced biomechanical analysis
- Athlete progress tracking
- Personalized training recommendations
- LLM-powered coaching feedback
- Cloud deployment
- Mobile application

---

##  Project Objective

The goal of this project is to demonstrate how **Computer Vision, Deep Learning, Machine Learning, and multimodal feature fusion** can be combined to create an automated sports performance assessment system.

Instead of simply identifying an action, the system aims to analyse **how the action was performed**, quantify relevant movement characteristics, generate a performance score, and provide actionable feedback.

---

