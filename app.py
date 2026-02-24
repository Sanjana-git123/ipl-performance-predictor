import requests
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="AI IPL Performance Intelligence",
    page_icon="🏏",
    layout="wide"
)
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("ai_cricket_bg.png");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* Dark overlay for readability */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 10, 25, 0.80);
        z-index: -1;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# =====================================================
# BACKGROUND + GLOBAL STYLING
import base64

def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_image = get_base64("ai_cricket_bg.png")

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bg_image}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 10, 25, 0.80);
        z-index: -1;
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
# CRICKET LIVE API
# =====================================================

CRIC_API_KEY = "79725bff-7883-4c9b-b37c-e6f6d7e75505" 

def fetch_player_live_data(player_name):

    url = "https://api.cricapi.com/v1/players?apikey=79725bff-7883-4c9b-b37c-e6f6d7e75505"
    params = {
        "apikey": CRIC_API_KEY,
        "offset": 0
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        return None

    data = response.json()

    for player in data.get("data", []):
        if player_name.lower() in player["name"].lower():
            return player

    return None


def build_live_features(api_player_data):

    features_dict = {}

    # TEMP mapping (adjust later)
    features_dict["Runs_Scored"] = api_player_data.get("runs", 0)
    features_dict["Year"] = 2026

    features_dict["Weather_Sunny"] = 1
    features_dict["Weather_Cloudy"] = 0
    features_dict["Weather_Humid"] = 0

    features_dict["Pitch_Flat"] = 1
    features_dict["Pitch_Green"] = 0
    features_dict["Pitch_Dusty"] = 0

    return pd.DataFrame([features_dict])

# =====================================================
# HERO SECTION
# =====================================================
st.markdown('<div class="hero-title">🏏 AI IPL Performance Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Predictive Modeling • Condition Optimization • Generative AI Insights</div>', unsafe_allow_html=True)
st.markdown('<div class="glow-line"></div>', unsafe_allow_html=True)

# =====================================================
# PLAYER SELECTION
# =====================================================
player_list = sorted(data["Player_Name"].unique())
selected_player = st.selectbox("Select Player", player_list)

# =====================================================
# OPTIMIZATION FUNCTION
# =====================================================
def optimize_conditions(player_df):

    weather_options = ["Sunny", "Cloudy", "Humid"]
    pitch_options = ["Flat", "Green", "Dusty"]

    best_score = -1
    best_weather = None
    best_pitch = None
    best_input_df = None

    for weather in weather_options:
        for pitch in pitch_options:

            temp_input = player_df.copy()

            for col in temp_input.columns:
                if col.startswith("Weather_") or col.startswith("Pitch_"):
                    temp_input[col] = 0

            if f"Weather_{weather}" in temp_input.columns:
                temp_input[f"Weather_{weather}"] = 1

            if f"Pitch_{pitch}" in temp_input.columns:
                temp_input[f"Pitch_{pitch}"] = 1

            input_df = temp_input[features]
            prediction = model.predict(input_df)[0]

            if prediction > best_score:
                best_score = prediction
                best_weather = weather
                best_pitch = pitch
                best_input_df = input_df

    return best_score, best_weather, best_pitch, best_input_df


# =====================================================
# MAIN LOGIC
# =====================================================
if selected_player:

    live_data = fetch_player_live_data(selected_player)

    if live_data is None:

        st.error("Live player data not found.")
    else:    
        latest_player = build_live_features(live_data)
        current_runs = latest_player["Runs_Scored"].iloc[0]
        predicted_runs, best_weather, best_pitch, best_input_df = optimize_conditions(latest_player)

        tree_predictions = np.array([
            tree.predict(best_input_df)[0]
            for tree in model.estimators_
        ])

        std_dev = np.std(tree_predictions)
        confidence = max(0, 100 - std_dev * 2)

        col1, col2 = st.columns([2, 1])

        # ================= LEFT: DASHBOARD =================
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)

            st.subheader("📊 Performance Dashboard")

            m1, m2, m3 = st.columns(3)
            m1.metric("Current Runs", int(current_runs))
            m2.metric("Predicted Runs", f"{int(predicted_runs)} ± {int(std_dev)}")
            m3.metric("Confidence", f"{confidence:.1f}%")

            st.info(f"Optimal Conditions → Weather: {best_weather}, Pitch: {best_pitch}")

            # Radar Chart
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

            # Trend
            st.subheader("📉 Historical Performance Trend")
            trend_df = player.sort_values("Year")[["Year", "Runs_Scored"]]
            st.line_chart(trend_df.set_index("Year"))

            st.markdown('</div>', unsafe_allow_html=True)

        # ================= RIGHT: AI PANEL =================
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)

            st.subheader("🤖 AI Insight Engine")

            delta = predicted_runs - current_runs

            if delta > 0:
                narrative = f"""
                {selected_player} shows strong upward scoring trajectory.
                Expected improvement: {int(delta)} runs.
                Model Confidence: {confidence:.1f}%.
                Conditions highly favorable for aggressive scoring.
                """
            else:
                narrative = f"""
                Performance volatility detected.
                Predicted decline: {abs(int(delta))} runs.
                Confidence Level: {confidence:.1f}%.
                """

            st.write(narrative)
            st.progress(int(confidence))

            st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# CHAT SECTION
# =====================================================
st.markdown('<div class="glow-line"></div>', unsafe_allow_html=True)
st.subheader("💬 AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask about player form, risk, conditions...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})

    ai_response = "Generative AI module will respond once OpenAI integration is enabled."

    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    st.rerun()