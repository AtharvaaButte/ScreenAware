# ScreenAware — Digital Diet & Mental Health Wellness Platform

An intelligent digital wellness platform that combines machine learning with behavioral habit tracking to detect digital burnout risk, predict daily mood ratings, and cluster screen usage personas.

---

## Overview

Modern digital lifestyles blur the line between productive device usage and chronic digital fatigue. **ScreenAware** provides actionable, data-driven wellness insights by analyzing daily behavioral telemetry—including screen time distribution, sleep duration, sleep quality, perceived stress, and physical exercise.

Rather than relying on arbitrary limits, the platform employs a production machine learning pipeline that runs real-time multi-task inference:
1. **Digital Burnout Risk Classification**: Categorizes user risk into `Low`, `Medium`, or `High` tiers using an ensemble gradient boosted classifier trained on a clinically motivated **Composite Digital Strain Index (CDSI)**.
2. **Mood Rating Regression**: Predicts user mood ratings on a standardized 1–5 scale using non-linear gradient boosting regression with interaction feature engineering.
3. **Usage Persona Clustering**: Unsupervised segmentation using K-Means (k = 5) to identify behavioral archetypes based on multi-dimensional screen time allocation.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       React + Vite                          │
│   Landing Page  •  Assessment Form  •  Analytics Dashboard   │
└──────────────┬──────────────────────────────▲───────────────┘
               │ JSON Telemetry               │ Live Report
               ▼                              │
┌─────────────────────────────────────────────┴───────────────┐
│                    FastAPI Backend (Port 8000)              │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │                  Live ML Inference                      │ │
│ │  • Risk Classifier (XGBoost)       -> 91% Test Accuracy │ │
│ │  • Mood Regressor (XGBoost)        -> 1–5 Scaled Mood   │ │
│ │  • Usage Persona (K-Means k=5)     -> 4D Clustering     │ │
│ └─────────────────────────────────────────────────────────┘ │
└──────────────┬──────────────────────────────▲───────────────┘
               │                              │
               ▼                              │ Historical Trends
┌─────────────────────────────┐  ┌────────────┴───────────────┐
│        Firebase Auth        │  │       MongoDB Database     │
│   User Identity & Sessions  │  │  Telemetry Logs & History  │
└─────────────────────────────┘  └────────────────────────────┘
```

---

## Machine Learning Methodology

### 1. Digital Burnout Risk Classification (XGBoost)
* **Features**: `daily_screen_time_hours`, `sleep_duration_hours`, `stress_level`, `sleep_quality`, `physical_activity_hours_per_week`.
* **Formulation**: Grounded in digital health literature, risk labels are derived via the **Composite Digital Strain Index (CDSI)**:

  ![CDSI Formula](https://fxtex.codecogs.com/svg.image?%5Ctext%7BCDSI%7D%3D0.40%20%5Ccdot%20%5Ctext%7BScore%7D_%7B%5Ctext%7Bdisp%7D%7D%2B0.40%20%5Ccdot%20%5Ctext%7BScore%7D_%7B%5Ctext%7Bstress%5C_sleep%7D%7D%2B0.20%20%5Ccdot%20%5Ctext%7BScore%7D_%7B%5Ctext%7Binact%7D%7D)

  * *Screen-to-Sleep Displacement*: Ratio of total screen time to restorative sleep.
  * *Stress & Sleep Quality Burden*: Interaction between perceived stress and inverted sleep quality.
  * *Inactivity Deficit*: Sedentary screen time offset by physical activity buffer.
* **Performance**:
  * **Holdout Accuracy**: **91.00%**
  * **5-Fold Cross-Validation Accuracy**: **90.37%**
  * **Precision / Recall**: High (0.94 / 0.80), Low (0.94 / 0.96), Medium (0.88 / 0.92).
  * **Primary Feature Drivers**: Daily screen time (34.0%), Sleep quality (19.2%), Stress level (17.6%), Sleep duration (16.7%), Physical exercise (12.6%).

### 2. Mood Rating Non-Linear Regression (XGBoost)
* **Target**: Continuous mood estimation dynamically mapped to a 1–5 rating scale.
* **Feature Engineering**:
  * `screen_sleep_ratio`: Screen time per hour of sleep.
  * `stress_x_sleep`: Interaction between stress level and sleep quality.
  * `activity_balance`: Physical activity normalized by screen exposure.
  * `wellness_score`: Net wellness buffer balancing sleep/activity against stress/screen load.
* **Performance**: Test MAE of 2.64 on raw scale (0.5 on mapped 1–5 scale).

### 3. Behavioral Usage Clustering (K-Means, k = 5)
* **Telemetry Analyzed**: `social_media_hours`, `gaming_hours`, `entertainment_hours`, `work_related_hours`.
* **Discovered Personas**:
  * **Social Media Dominant** (3.5h social): Heavy mobile scrolling lifestyle.
  * **Entertainment Focused** (4.0h streaming): High streaming & OTT consumption.
  * **Gaming Heavy** (2.4h gaming): Dedicated gaming habits.
  * **Work Oriented** (2.6h work, <0.6h gaming): Productive desktop-centric usage.
  * **Casual Leisure** (2.7h leisure, <0.7h work): Low-work relaxation usage.

---

## Project Structure

```
ScreenAware/
├── backend/
│   ├── app/
│   │   ├── models/                # Production model pickles & Pydantic schemas
│   │   │   ├── risk_model.pkl     # Trained XGBoost Risk Classifier
│   │   │   ├── mood_model.pkl     # Trained XGBoost Mood Regressor
│   │   │   ├── cluster_model.pkl  # Trained KMeans Clustering Model
│   │   │   ├── scaler_*.pkl       # StandardScalers for features
│   │   │   ├── artifacts.pkl      # Feature definitions & persona mappings
│   │   │   └── user_data.py       # Pydantic v2 schemas & MongoDB connection
│   │   ├── routers/
│   │   │   └── analytics.py       # MongoDB analytics endpoints
│   │   └── main.py                # FastAPI app & prediction endpoints
│   └── requirements.txt           # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── LandingPage.jsx    # Hero section & navigation
│   │   │   ├── LoginForm.jsx      # Firebase auth login
│   │   │   ├── SignupForm.jsx     # Firebase auth registration
│   │   │   ├── Home.jsx           # Dashboard overview
│   │   │   ├── DataForm.jsx       # Digital habit assessment form
│   │   │   ├── Report.jsx         # AI prediction report card
│   │   │   ├── Analytics.jsx      # Recharts analytics dashboard
│   │   │   └── TipsPage.jsx       # Personalized wellness recommendations
│   │   ├── App.jsx                # Main router & state management
│   │   └── firebase.js            # Firebase client SDK initialization
│   └── package.json
├── ml_training/
│   ├── data/
│   │   └── digital_diet_mental_health.csv  # Training dataset (2,000 records)
│   ├── experiments/                        # Archived R&D notebooks (ml1 - ml4)
│   │   ├── ml.ipynb
│   │   ├── ml2.ipynb
│   │   ├── ml3.ipynb
│   │   └── ml4.ipynb
│   ├── models/                             # Master generated model artifacts
│   ├── train_pipeline.py                   # Master reproducible CLI training script
│   └── model_training.ipynb                # Clean, fully executed Jupyter notebook
└── README.md
```

---

## Tech Stack

* **Frontend**: React 19, Vite, Recharts, Framer Motion, Firebase Client SDK
* **Backend**: FastAPI, Pydantic v2, Uvicorn, PyMongo
* **Machine Learning**: Python 3.14, XGBoost, Scikit-Learn, Pandas, NumPy, Joblib
* **Database**: MongoDB
* **Authentication**: Firebase Authentication (Email/Password & Google Sign-In)

---

## Installation & Setup

### 1. Prerequisites
* **Python**: 3.11+ (Fully tested and verified on Python 3.14)
* **Node.js**: v18+ (Tested on Node v24)
* **MongoDB**: Running locally on `localhost:27017` (Optional for live ML demos)

---

### 2. Backend Setup
1. Open a terminal and navigate to the project root:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
2. Start the FastAPI server:
   ```bash
   uvicorn backend.app.main:app --reload --port 8000
   ```
   * Backend will be active at: `http://localhost:8000`
   * Interactive Swagger API docs: `http://localhost:8000/docs`

---

### 3. Frontend Setup
1. In a second terminal, navigate to the `frontend` folder:
   ```bash
   cd frontend
   npm install
   ```
2. Start the Vite development server:
   ```bash
   npm start
   ```
   * Frontend will launch at: `http://localhost:5173`

---

### 4. Retraining the ML Pipeline (Optional)
To retrain all models from scratch and regenerate production artifacts:
```bash
python ml_training/train_pipeline.py
```
This updates all `.pkl` models and exports them directly into `backend/app/models/`.

---

## API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/predict_report` | Runs multi-task inference: returns risk level, mood rating, cluster persona, and dominant category. |
| `POST` | `/predict_risk` | Predicts digital burnout risk tier (`Low`, `Medium`, `High`). |
| `POST` | `/predict_mood` | Returns predicted mood rating on 1–5 scale. |
| `POST` | `/predict_cluster` | Returns behavioral usage cluster persona. |
| `POST` | `/api/user-data` | Persists user assessment and predictions to MongoDB. |
| `GET` | `/api/analytics/overview/{user_id}` | Aggregates 30-day screen time, mood averages, and risk distribution. |
| `GET` | `/api/analytics/detailed/{user_id}` | Returns daily and weekly telemetry breakdowns for charting. |
| `GET` | `/health` | Server and model health check status. |

---

## Demonstration Guide

1. Start both backend and frontend using the commands above.
2. Navigate to `http://localhost:5173` and click **Assessment** in the navigation bar.
3. Enter habit telemetry (e.g., Screen Time: `8.5h`, Sleep: `5.0h`, Stress: `8`, Quality: `3`, Social: `4h`, Gaming: `2h`, Entertainment: `1.5h`, Work: `1h`).
4. Click **Generate Report** to trigger live inference across all three ML models.
5. Inspect the generated report displaying **Risk Level**, **Mood Rating**, **Usage Persona**, and **Personalized Recommendations**.

---

## Author

**Atharva Butte**  
Digital Wellness & Machine Learning Platform
