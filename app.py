import requests
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import base64
import os


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="IPL Performance Predictor",
    page_icon="🏏",
    layout="wide"
)

# =====================================================
# BACKGROUND + FUTURISTIC CSS
# =====================================================
def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_image = get_base64("ai_cricket_bg.png")

st.markdown(
    f"""
    <style>

    /* ========== BACKGROUND IMAGE ========== */
    .stApp {{
        background-image: url("data:image/png;base64,{bg_image}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    /* Dark overlay (FIXED layering) */
    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        background: rgba(5, 15, 30, 0.75);
        z-index: 0;
    }}

    /* Ensure all content appears above overlay */
    .stApp > * {{
        position: relative;
        z-index: 1;
    }}

    /* ========== HERO TITLE ========== */
    .hero-title {{
        font-size: 52px;
        font-weight: bold;
        text-align: center;
        color: #00f7ff;
        text-shadow: 0 0 10px #00f7ff,
                     0 0 20px #00f7ff,
                     0 0 40px #00f7ff;
        margin-top: 20px;
        animation: glow 2s ease-in-out infinite alternate;
    }}

    @keyframes glow {{
        from {{ text-shadow: 0 0 10px #00f7ff; }}
        to {{ text-shadow: 0 0 25px #00f7ff; }}
    }}

    .hero-subtitle {{
        font-size: 18px;
        text-align: center;
        color: #d1faff;
        margin-bottom: 10px;
    }}

    .glow-line {{
        height: 3px;
        background: linear-gradient(90deg, transparent, #00f7ff, transparent);
        margin: 15px 0;
        box-shadow: 0 0 15px #00f7ff;
    }}

    /* ========== GLASS CARD (IMPROVED VISIBILITY) ========== */
    .card {{
        background: rgba(10, 20, 40, 0.88);
        backdrop-filter: blur(12px);
        padding: 22px;
        border-radius: 18px;
        border: 1px solid rgba(0,255,255,0.35);
        box-shadow: 0 0 30px rgba(0,255,255,0.25);
        margin-bottom: 20px;
    }}

    /* ========== DASHBOARD STATS ========== */
    .dashboard-label {{
        font-size: 14px;
        font-weight: 600;
        color: #cbd5e1;
        letter-spacing: 0.5px;
    }}

    .dashboard-value {{
        font-size: 32px;
        font-weight: 700;
        color: #ffffff;
        margin-top: 6px;
    }}

    /* ========== AI INSIGHT TEXT ========== */
    .insight-text {{
        font-size: 16px;
        font-weight: 600;
        color: #ffffff;
        line-height: 1.6;
    }}

    /* ========== SELECT DROPDOWN FIX ========== */
    div[data-baseweb="select"] > div {{
        background-color: rgba(0, 30, 60, 0.9) !important;
        color: white !important;
        border: 1px solid #00f7ff !important;
        border-radius: 8px !important;
    }}

    /* Fix dropdown text color */
    div[data-baseweb="select"] span {{
        color: white !important;
    }}

    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_assets():
    model = joblib.load("updated_model.pkl")
    data = joblib.load("updated_processed_data.pkl")
    features = joblib.load("updated_features.pkl")
    return model, data, features

model, data, features = load_assets()



# =====================================================
# HERO
# =====================================================
st.markdown('<div class="hero-title">🏏 IPL Performance Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Predictive Modeling • Condition Optimization</div>', unsafe_allow_html=True)
st.markdown('<div class="glow-line"></div>', unsafe_allow_html=True)

# =====================================================
# PLAYER SELECTION
# =====================================================
player_list = sorted(data["Player_Name"].unique())
selected_player = st.selectbox("Select Player", player_list)


# =====================================================
# MAIN
# =====================================================
if selected_player:

    player = data[data["Player_Name"] == selected_player]

   
    # ===========================
    # BUILD MODEL INPUT
    # ===========================
    
    latest_full = player.sort_values("Year").tail(1).copy()
    
    st.success("AI Model analyzing latest seasonal data")

    current_runs = latest_full["Runs_Scored"].iloc[0]

    model_input = latest_full.reindex(columns=features, fill_value=0)

    
    # -------------------------
    # PREDICTION
    # -------------------------
    predicted_runs = model.predict(model_input.values)[0]

    tree_predictions = np.array([
        tree.predict(model_input.values)[0]
        for tree in model.estimators_
    ])

    std_dev = np.std(tree_predictions)
    
    # ===========================
# AUTO OPTIMAL CONDITIONS
# ===========================

# Weather Conditions
    weather_conditions = ["Sunny", "Cloudy", "Humid"]
    weather_adjustments = [15, 10, 5]

    weather_scores = [predicted_runs + adj for adj in weather_adjustments]
    best_weather = weather_conditions[np.argmax(weather_scores)]

# Pitch Conditions
    pitch_types = ["Flat", "Green", "Dusty"]
    pitch_adjustments = [20, 10, 15]

    pitch_scores = [predicted_runs + adj for adj in pitch_adjustments]
    best_pitch = pitch_types[np.argmax(pitch_scores)]

# Calculate boosted predicted score
    optimal_score = max(weather_scores) + max(pitch_adjustments)
    # ===========================
    # DASHBOARD
    # ===========================
    col1, col2 = st.columns([2, 1])

    # LEFT PANEL
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("📊 Performance Dashboard")

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Current Runs", int(current_runs))
        m2.metric("Predicted Runs", f"{int(predicted_runs)} ± {int(std_dev)}")
        m3.metric("Current Avg", f"{current_avg:.2f}")
        m4.metric(
            "Predicted Avg",
            f"{predicted_avg:.2f}",
            delta=f"{predicted_avg - current_avg:.2f}"
        )

        importance = model.feature_importances_

        feat_imp = pd.DataFrame({
            "Feature": features,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False).head(6)

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=feat_imp["Importance"],
            theta=feat_imp["Feature"],
            fill='toself'
        ))

        fig.update_layout(
            polar=dict(bgcolor="#001a2d"),
            paper_bgcolor="#001a2d",
            font_color="white"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📉 Historical Performance Trend")
        trend_df = player.sort_values("Year")[["Year", "Runs_Scored"]]
        st.line_chart(trend_df.set_index("Year"))

        st.markdown('</div>', unsafe_allow_html=True)

    # RIGHT PANEL
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("🤖 AI Insight Engine")

        delta = predicted_runs - current_runs

        if delta > 0:
            narrative = f"{selected_player} is projected to improve by {int(delta)} runs."
        else:
            narrative = f"{selected_player} may decline by {abs(int(delta))} runs."

        st.write(narrative)
        st.progress(int(confidence))
        st.markdown("### 🌦 Optimal Match Conditions")

        st.write(f"**Best Weather:** {best_weather}")
        st.write(f"**Best Pitch:** {best_pitch}")
        st.write(f"**Projected Runs in Optimal Conditions:** {int(optimal_score)}")
        st.markdown('</div>', unsafe_allow_html=True)
    
# =====================================================
