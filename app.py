import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import xgboost as xgb

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Neural AI Student Predictor v3.0",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CYBERPUNK CSS STYLING
# ============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --cyber-bg: #050816;
    --cyber-bg-dark: #0D1117;
    --cyber-glass: rgba(255, 255, 255, 0.03);
    --cyber-glow-cyan: #00D4FF;
    --cyber-glow-purple: #A855F7;
    --cyber-glow-pink: #EC4899;
    --cyber-gradient: linear-gradient(135deg, #00D4FF 0%, #A855F7 40%, #EC4899 70%, #F472B6 100%);
    --text-primary: #E6EDF3;
    --text-secondary: #8B949E;
}

* { font-family: 'Inter', sans-serif; }

body, .main {
    background: var(--cyber-bg);
    background-attachment: fixed;
}

#neural-bg {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    z-index: -1;
    background:
        radial-gradient(circle at 20% 80%, rgba(120,119,198,0.3) 0%, transparent 50%),
        radial-gradient(circle at 80% 20%, rgba(255,119,198,0.3) 0%, transparent 50%),
        radial-gradient(circle at 40% 40%, rgba(120,219,255,0.3) 0%, transparent 50%),
        linear-gradient(60deg, #00D4FF33 0%, #A855F733 30%, transparent 70%);
    animation: cyberWave 20s ease-in-out infinite;
}

@keyframes cyberWave {
    0%, 100% { transform: scale(1) rotate(0deg); opacity: 0.5; }
    50%       { transform: scale(1.1) rotate(180deg); opacity: 0.8; }
}

.cyber-card {
    background: var(--cyber-glass);
    backdrop-filter: blur(25px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 24px;
    box-shadow: 0 25px 50px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.1);
    position: relative;
    overflow: hidden;
    transition: all 0.5s cubic-bezier(0.23, 1, 0.32, 1);
    padding: 32px;
}

.cyber-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: var(--cyber-gradient);
    transform: scaleX(0);
    transition: transform 0.5s ease;
}

.cyber-card:hover {
    transform: translateY(-8px) scale(1.01);
    box-shadow: 0 40px 80px rgba(0,0,0,0.6), 0 0 40px rgba(0,212,255,0.3);
}

.cyber-card:hover::before { transform: scaleX(1); }

.cyber-title {
    font-family: 'Orbitron', monospace;
    font-weight: 900;
    background: var(--cyber-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.stSidebar {
    background: linear-gradient(180deg, var(--cyber-bg-dark) 0%, rgba(10,15,30,0.95) 100%);
    border-right: 1px solid rgba(255,255,255,0.1);
}

input:focus, select:focus {
    border-color: var(--cyber-glow-cyan) !important;
    box-shadow: 0 0 20px rgba(0,212,255,0.4) !important;
}
</style>

<div id="neural-bg"></div>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS  (defined BEFORE they are called anywhere)
# ============================================================================

GRADE_MAP = {0: 'A 🎓', 1: 'B 📚', 2: 'C ⚠️', 3: 'F ❌'}
PARTICIPATION_MAP = {'Low': 0, 'Medium': 1, 'High': 2}


def feature_engineering(student: dict) -> dict:
    """Derive summary statistics from a student record."""
    total_score = (
        student['assignment_score'] + student['midterm_score'] + student['final_score']
    ) / 3

    att = student['attendance']
    if att >= 90:
        attendance_cat = 'Excellent'
    elif att >= 80:
        attendance_cat = 'Good'
    elif att >= 70:
        attendance_cat = 'Average'
    else:
        attendance_cat = 'Poor'

    # Cap study_hours contribution so index stays in 0-100
    study_contrib = min(student['study_hours'] / 40 * 100, 100)
    performance_index = min(
        total_score * 0.5 + student['attendance'] * 0.3 + study_contrib * 0.2,
        100.0
    )
    risk_score = max(100 - performance_index, 0)

    return {
        'Total Score': f"{total_score:.1f}%",
        'Attendance Category': attendance_cat,
        'Performance Index': f"{performance_index:.1f}/100",
        'Risk Score': f"{risk_score:.0f}",
    }


def get_prediction(student_data: dict):
    """Return grade predictions and confidence % for every trained model."""
    features = np.array([[
        student_data['attendance'],
        student_data['assignment_score'],
        student_data['midterm_score'],
        student_data['final_score'],
        student_data['study_hours'],
        PARTICIPATION_MAP[student_data['participation']],
        PARTICIPATION_MAP[student_data['discipline']],
        PARTICIPATION_MAP[student_data['engagement']],
        student_data['late_submissions'],
    ]])

    predictions = {}
    probabilities = {}

    for name, model in st.session_state.models.items():
        pred = model.predict(features)[0]
        probs = model.predict_proba(features)[0]
        predictions[name] = GRADE_MAP[pred]
        probabilities[name] = float(np.max(probs) * 100)

    return predictions, probabilities


# ============================================================================
# MODEL TRAINING  (grade column excluded from X to prevent data leakage)
# ============================================================================
@st.cache_data
def train_models():
    np.random.seed(42)
    n_samples = 2000

    # Features
    attendance        = np.random.uniform(0, 100, n_samples)   # full range → matches slider
    assignment_score  = np.random.uniform(0, 100, n_samples)
    midterm_score     = np.random.uniform(0, 100, n_samples)
    final_score       = np.random.uniform(0, 100, n_samples)
    study_hours       = np.random.uniform(0, 40,  n_samples)
    participation     = np.random.choice([0, 1, 2], n_samples)
    discipline        = np.random.choice([0, 1, 2], n_samples)
    engagement        = np.random.choice([0, 1, 2], n_samples)
    late_submissions  = np.clip(np.random.poisson(3, n_samples), 0, 25)

    # Label (NOT included in X)
    total_score = (assignment_score + midterm_score + final_score) / 3
    grade = np.where(total_score >= 80, 0,
            np.where(total_score >= 65, 1,
            np.where(total_score >= 50, 2, 3)))

    X = pd.DataFrame({
        'attendance':       attendance,
        'assignment_score': assignment_score,
        'midterm_score':    midterm_score,
        'final_score':      final_score,
        'study_hours':      study_hours,
        'participation':    participation,
        'discipline':       discipline,
        'engagement':       engagement,
        'late_submissions': late_submissions,
    })
    y = grade

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_defs = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree':       DecisionTreeClassifier(random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42),
        'XGBoost':             xgb.XGBClassifier(random_state=42, eval_metric='mlogloss'),
    }

    trained_models = {}
    results = {}

    for name, model in model_defs.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results[name] = {
            'accuracy':         float(accuracy_score(y_test, y_pred)),
            'precision':        float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'recall':           float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            'f1':               float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
        }
        trained_models[name] = model

    return trained_models, results

# ============================================================================
# SESSION STATE
# ============================================================================
if 'current_student' not in st.session_state:
    st.session_state.current_student = None
if 'students' not in st.session_state:
    st.session_state.students = []
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Train once
if not st.session_state.models_trained:
    with st.spinner("🧠 Training Neural Network Models..."):
        st.session_state.models, st.session_state.model_results = train_models()
    st.session_state.models_trained = True

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 32px 24px;'>
        <h1 class='cyber-title' style='font-size: 28px; margin: 0;'>🧠 Neural AI</h1>
        <p style='color: var(--text-secondary); font-size: 13px; margin: 8px 0 0 0;'>
            Student Performance v3.0
        </p>
    </div>
    """, unsafe_allow_html=True)

    pages = {
        "🌌 Neural Core":    "home",
        "👤 Student Info":   "student_info",
        "🔮 Predictions":    "predictions",
        "📊 Deep Analysis":  "analysis",
        "⚡ Model Insights": "models",
    }

    for label, key in pages.items():
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.page = key
            st.rerun()

    st.markdown("---")

    # Save current student
    if st.session_state.current_student:
        if st.button("💾 Save Student", use_container_width=True):
            st.session_state.students.append(
                st.session_state.current_student.copy()
            )
            st.success("✅ Student saved!")

    # Download saved students as JSON
    if st.session_state.students:
        st.download_button(
            label="📥 Download Records",
            data=json.dumps(st.session_state.students, indent=2),
            file_name="students.json",
            mime="application/json",
            use_container_width=True,
        )

    # Load students from JSON
    uploaded = st.file_uploader("📁 Load Students (JSON)", type=["json"])
    if uploaded:
        try:
            loaded = json.load(uploaded)
            if isinstance(loaded, list):
                st.session_state.students = loaded
                st.success(f"📥 Loaded {len(loaded)} student(s)!")
            else:
                st.error("Invalid file format — expected a JSON list.")
        except Exception as e:
            st.error(f"Could not read file: {e}")

# ============================================================================
# PAGE ROUTING
# ============================================================================
page = st.session_state.page

# ── HOME ─────────────────────────────────────────────────────────────────────
if page == "home":
    st.markdown("""
    <div style='text-align: center; padding: 80px 40px;'>
        <h1 class='cyber-title' style='font-size: 64px; margin-bottom: 32px;'>
            Neural AI Predictor
        </h1>
        <div style='font-size: 22px; color: var(--text-primary); margin-bottom: 64px;'>
            <span style='color: var(--cyber-glow-cyan);'>Quantum Neural Analysis</span>
            of Student Performance
        </div>
        <div style='display: flex; justify-content: center; gap: 32px; flex-wrap: wrap;'>
            <div class="cyber-card" style='padding: 48px 64px; min-width: 240px;'>
                <h3 style='color: var(--cyber-glow-cyan); font-size: 44px; margin: 0 0 12px 0;'>4</h3>
                <p style='color: var(--text-secondary); font-size: 16px; margin: 0;'>AI Models</p>
            </div>
            <div class="cyber-card" style='padding: 48px 64px; min-width: 240px;'>
                <h3 style='color: var(--cyber-glow-purple); font-size: 44px; margin: 0 0 12px 0;'>Real-time</h3>
                <p style='color: var(--text-secondary); font-size: 16px; margin: 0;'>Predictions</p>
            </div>
            <div class="cyber-card" style='padding: 48px 64px; min-width: 240px;'>
                <h3 style='color: var(--cyber-glow-pink); font-size: 44px; margin: 0 0 12px 0;'>A/B/C/F</h3>
                <p style='color: var(--text-secondary); font-size: 16px; margin: 0;'>Grade Matrix</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── STUDENT INFO ──────────────────────────────────────────────────────────────
elif page == "student_info":
    st.markdown(
        "<h1 class='cyber-title' style='font-size: 48px;'>👤 Student Information</h1>",
        unsafe_allow_html=True,
    )

    with st.form("student_form"):
        col1, col2 = st.columns(2)

        with col1:
            name        = st.text_input("👤 Full Name")
            student_id  = st.text_input("🆔 Student ID")
            age         = st.number_input("🎂 Age", 15, 35, 20)
            gender      = st.selectbox("⚧️ Gender", ["Male", "Female", "Other"])
            email       = st.text_input("📧 Email")

        with col2:
            phone           = st.text_input("📱 Phone")
            attendance      = st.slider("📊 Attendance %", 0.0, 100.0, 85.0, 1.0)
            assignment_score = st.slider("📝 Assignment Score", 0.0, 100.0, 75.0, 1.0)
            midterm_score   = st.slider("📚 Midterm Score", 0.0, 100.0, 70.0, 1.0)
            final_score     = st.slider("🎯 Final Score", 0.0, 100.0, 65.0, 1.0)

        col3, col4, col5 = st.columns(3)
        with col3:
            study_hours  = st.slider("⏰ Study Hours/Week", 0.0, 40.0, 15.0, 0.5)
        with col4:
            participation = st.selectbox("🤝 Participation", ["Low", "Medium", "High"])
        with col5:
            discipline   = st.selectbox("🎯 Discipline", ["Low", "Medium", "High"])

        col6, col7, _ = st.columns(3)
        with col6:
            engagement       = st.selectbox("💡 Engagement", ["Low", "Medium", "High"])
        with col7:
            late_submissions = st.number_input("⏳ Late Submissions", 0, 25, 2)

        submitted = st.form_submit_button("🔮 Get AI Prediction", use_container_width=True)

    # Process AFTER the form closes (outside the with block)
    if submitted:
        # Basic validation
        errors = []
        if not name.strip():
            errors.append("Full Name is required.")
        if not student_id.strip():
            errors.append("Student ID is required.")
        if email.strip() and "@" not in email:
            errors.append("Email address looks invalid.")

        if errors:
            for err in errors:
                st.error(err)
        else:
            st.session_state.current_student = {
                'name':             name.strip(),
                'student_id':       student_id.strip(),
                'age':              age,
                'gender':           gender,
                'email':            email.strip(),
                'phone':            phone.strip(),
                'attendance':       attendance,
                'assignment_score': assignment_score,
                'midterm_score':    midterm_score,
                'final_score':      final_score,
                'study_hours':      study_hours,
                'participation':    participation,
                'discipline':       discipline,
                'engagement':       engagement,
                'late_submissions': int(late_submissions),
                'timestamp':        datetime.now().isoformat(),
            }
            st.session_state.page = "predictions"
            st.rerun()

# ── PREDICTIONS ───────────────────────────────────────────────────────────────
elif page == "predictions":
    if not st.session_state.current_student:
        st.warning("⚠️ No student data found. Please fill in Student Info first.")
        st.stop()

    st.markdown(
        "<h1 class='cyber-title' style='font-size: 48px;'>🔮 AI Predictions</h1>",
        unsafe_allow_html=True,
    )

    predictions, probabilities = get_prediction(st.session_state.current_student)
    best_model = max(probabilities, key=probabilities.get)
    best_pred  = predictions[best_model]
    best_conf  = probabilities[best_model]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="cyber-card" style="padding: 48px; text-align: center;">
            <div style='font-size: 72px; margin-bottom: 16px;'>{best_pred}</div>
            <div style='font-size: 44px; font-weight: 800; color: var(--cyber-glow-cyan);'>
                {best_conf:.1f}%
            </div>
            <div style='margin-top: 16px; padding: 10px 20px; border-radius: 16px;
                        background: rgba(0,212,255,0.15);
                        border: 1px solid var(--cyber-glow-cyan);
                        font-size: 16px; font-weight: 700;'>
                🏆 Best model: {best_model}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Bar chart — confidence per model (more meaningful than a pie)
        fig = go.Figure(go.Bar(
            x=list(probabilities.keys()),
            y=list(probabilities.values()),
            marker_color=['#00D4FF', '#A855F7', '#EC4899', '#FBBF24'],
            text=[f"{v:.1f}%" for v in probabilities.values()],
            textposition='outside',
        ))
        fig.update_layout(
            yaxis_title="Confidence (%)",
            yaxis_range=[0, 110],
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#E6EDF3',
            height=380,
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Summary table
    pred_df = pd.DataFrame({
        'Model':      list(predictions.keys()),
        'Prediction': list(predictions.values()),
        'Confidence': [f"{p:.1f}%" for p in probabilities.values()],
    })
    st.dataframe(pred_df, use_container_width=True, hide_index=True)

# ── DEEP ANALYSIS ─────────────────────────────────────────────────────────────
elif page == "analysis":
    if not st.session_state.current_student:
        st.warning("⚠️ No student data found. Please fill in Student Info first.")
        st.stop()

    st.markdown(
        "<h1 class='cyber-title' style='font-size: 48px;'>📊 Deep Analysis</h1>",
        unsafe_allow_html=True,
    )

    student = st.session_state.current_student
    derived = feature_engineering(student)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="cyber-card">
            <h3 style='color: var(--cyber-glow-cyan); margin-top: 0;'>👤 Personal Details</h3>
            <p><strong>Name:</strong> {student.get('name', 'N/A')}</p>
            <p><strong>ID:</strong> {student.get('student_id', 'N/A')}</p>
            <p><strong>Age:</strong> {student.get('age', 'N/A')} &nbsp;|&nbsp;
               <strong>Gender:</strong> {student.get('gender', 'N/A')}</p>
            <p><strong>Email:</strong> {student.get('email', 'N/A')}</p>
            <p><strong>Phone:</strong> {student.get('phone', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        rows = "".join(
            f"<p><strong>{k}:</strong> "
            f"<span style='color: var(--cyber-glow-cyan);'>{v}</span></p>"
            for k, v in derived.items()
        )
        st.markdown(f"""
        <div class="cyber-card">
            <h3 style='color: var(--cyber-glow-purple); margin-top: 0;'>📈 Derived Features</h3>
            {rows}
        </div>
        """, unsafe_allow_html=True)

    # Feature importance (Random Forest)
    rf_model    = st.session_state.models['Random Forest']
    importances = rf_model.feature_importances_
    feat_names  = [
        'Attendance', 'Assignment', 'Midterm', 'Final',
        'Study Hours', 'Participation', 'Discipline', 'Engagement', 'Late Subs',
    ]
    top_factors = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)[:5]

    rows_html = "".join(
        f"""<div style='padding: 14px; margin: 8px 0;
                        background: rgba(0,212,255,0.08);
                        border-radius: 12px;
                        border-left: 4px solid var(--cyber-glow-cyan);'>
                <strong>📊 {factor}</strong> — Importance: {importance*100:.1f}%
            </div>"""
        for factor, importance in top_factors
    )
    st.markdown(f"""
    <div class="cyber-card" style='margin-top: 24px;'>
        <h3 style='color: var(--cyber-glow-pink); margin-top: 0;'>🧠 AI Insights — Top Features</h3>
        {rows_html}
    </div>
    """, unsafe_allow_html=True)

# ── MODEL INSIGHTS ────────────────────────────────────────────────────────────
elif page == "models":
    st.markdown(
        "<h1 class='cyber-title' style='font-size: 48px;'>⚡ Model Insights</h1>",
        unsafe_allow_html=True,
    )

    results = st.session_state.model_results

    # Store as floats so highlight_max works correctly
    metrics_df = pd.DataFrame({
        'Model':     list(results.keys()),
        'Accuracy':  [r['accuracy']  for r in results.values()],
        'Precision': [r['precision'] for r in results.values()],
        'Recall':    [r['recall']    for r in results.values()],
        'F1-Score':  [r['f1']        for r in results.values()],
    })

    # Format display only (keep underlying values as floats for highlight_max)
    display_df = metrics_df.copy()
    for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
        display_df[col] = display_df[col].map(lambda v: f"{v:.4f}")

    st.dataframe(
        metrics_df.style.highlight_max(
            subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            axis=0,
            color='#1a4a2e',          # dark green — visible on both light/dark
        ).format({c: '{:.4f}' for c in ['Accuracy', 'Precision', 'Recall', 'F1-Score']}),
        use_container_width=True,
        hide_index=True,
    )

    # Grouped bar chart
    fig = px.bar(
        metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Score'),
        x='Model', y='Score', color='Metric', barmode='group',
        color_discrete_sequence=['#00D4FF', '#A855F7', '#EC4899', '#FBBF24'],
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#E6EDF3',
        yaxis_range=[0, 1.05],
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Confusion matrix for the best model
    best_name = max(results, key=lambda k: results[k]['accuracy'])
    cm = results[best_name]['confusion_matrix']

    fig2 = px.imshow(
        cm,
        text_auto=True,
        aspect="auto",
        title=f"Confusion Matrix — {best_name}",
        color_continuous_scale='Blues',
        labels=dict(x="Predicted", y="Actual"),
        x=['A', 'B', 'C', 'F'],
        y=['A', 'B', 'C', 'F'],
    )
    fig2.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#E6EDF3',
    )
    st.plotly_chart(fig2, use_container_width=True)
