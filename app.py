import requests
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import base64
import os

# =====================================================
# RAPID API CONFIG
# =====================================================
RAPID_API_KEY = os.getenv("RAPID_API_KEY")

RAPID_API_HOST = "cricket-live-line-advance.p.rapidapi.com"

def fetch_match_data(match_id):
    if not RAPID_API_KEY:
        return None

    url = f"https://{RAPID_API_HOST}/match/{match_id}/innings"

    headers = {
        "X-RapidAPI-Key": RAPID_API_KEY,
        "X-RapidAPI-Host": RAPID_API_HOST
    }

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        st.error(f"API Error: {response.status_code}")
        return None

    return response.json()

def extract_player_live_runs(api_json, player_name):
    if not api_json:
        return None

    players = api_json["response"].get("players", [])
    innings = api_json["response"].get("innings", [])

    # Map player_id → name
    player_map = {p["player_id"]: p["name"] for p in players}

    for inning in innings:
        for fow in inning.get("fows", []):
            pid = fow["batsman_id"]
            name = player_map.get(pid, "")

            if player_name.lower() in name.lower():
                return fow["runs"]

    return None

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="AI IPL Performance Intelligence",
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
        background: rgba(0, 10, 25, 0.88);
        z-index: -1;
    }}

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
        color: #b3faff;
        margin-bottom: 10px;
    }}

    .glow-line {{
        height: 3px;
        background: linear-gradient(90deg, transparent, #00f7ff, transparent);
        margin: 15px 0;
        box-shadow: 0 0 15px #00f7ff;
    }}

    .card {{
        background: rgba(0, 25, 50, 0.65);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 18px;
        border: 1px solid rgba(0,255,255,0.4);
        box-shadow: 0 0 25px rgba(0,255,255,0.3);
    }}

    div[data-baseweb="select"] > div {{
        background-color: rgba(0, 30, 60, 0.8) !important;
        color: white !important;
        border: 1px solid #00f7ff !important;
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
st.markdown('<div class="hero-title">🏏 AI IPL Performance Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Predictive Modeling • Condition Optimization • Generative AI Insights</div>', unsafe_allow_html=True)
st.markdown('<div class="glow-line"></div>', unsafe_allow_html=True)

# =====================================================
# PLAYER SELECTION
# =====================================================
player_list = sorted(data["Player_Name"].unique())
selected_player = st.selectbox("Select Player", player_list)

# =====================================================
# OPTIMIZATION
# =====================================================
def optimize_conditions(player_df):
    best_score = -1
    best_input_df = None

    input_df = player_df[features]
    prediction = model.predict(input_df)[0]

    best_score = prediction
    best_input_df = input_df

    return best_score, best_input_df

# =====================================================
# MAIN
# =====================================================
if selected_player:

    player = data[data["Player_Name"] == selected_player]

    # ===========================
    # FETCH LIVE DATA FROM RAPIDAPI
    # ===========================

    match_id = 56789  # 🔥 Replace with real match_id
    api_data = fetch_match_data(match_id)
    live_runs = extract_player_live_runs(api_data, selected_player)

    # ===========================
    # BUILD MODEL INPUT
    # ===========================

    if live_runs is not None:
        latest_full = player.sort_values("Year").tail(1).copy()
        latest_full["Runs_Scored"] = live_runs
    else:
        latest_full = player.sort_values("Year").tail(1)

    latest_full = latest_full[features]

    current_runs = latest_full["Runs_Scored"].iloc[0]

    predicted_runs, best_input_df = optimize_conditions(latest_full)

    tree_predictions = np.array([
        tree.predict(best_input_df)[0]
        for tree in model.estimators_
    ])

    std_dev = np.std(tree_predictions)
    confidence = max(0, 100 - std_dev * 2)
    

# Extract current runs safely
    if "Runs_Scored" in latest_full.columns:
        current_runs = latest_full["Runs_Scored"].iloc[0]
    else:
        current_runs = 0

# Prepare model input separately
    model_input = latest_full[features]

    predicted_runs, best_input_df = optimize_conditions(model_input)
    

    tree_predictions = np.array([
        tree.predict(best_input_df)[0]
        for tree in model.estimators_
    ])

    std_dev = np.std(tree_predictions)
    confidence = max(0, 100 - std_dev * 2)

    col1, col2 = st.columns([2, 1])

    # LEFT PANEL
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("📊 Performance Dashboard")

        m1, m2, m3 = st.columns(3)
        m1.metric("Current Runs", int(current_runs))
        m2.metric("Predicted Runs", f"{int(predicted_runs)} ± {int(std_dev)}")
        m3.metric("Confidence", f"{confidence:.1f}%")

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

        st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# CHAT
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
    ai_response = "OpenAI integration will activate once API key is added."
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    st.rerun()