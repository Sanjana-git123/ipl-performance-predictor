import streamlit as st
import streamlit as st
st.title("🚀 THIS IS DEFINITELY APP2 🚀")
import joblib
import pandas as pd
import numpy as np

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="IPL Performance Predictor",
    page_icon="🏏",
    layout="wide"
)

# -----------------------------
# LOAD MODEL & DATA
# -----------------------------
model = joblib.load("updated_model.pkl")
data = joblib.load("updated_processed_data.pkl")
features = joblib.load("updated_features.pkl")

# -----------------------------
# TITLE
# -----------------------------
st.title("🏏 IPL Player Performance Forecast (2026)")
st.markdown("Advanced Machine Learning based prediction system")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("🔍 Select Player & Conditions")

player_list = sorted(data["Player_Name"].unique())
selected_player = st.sidebar.selectbox("Player", player_list)

weather_input = st.sidebar.selectbox("Weather", ["Sunny", "Cloudy", "Humid"])
pitch_input = st.sidebar.selectbox("Pitch", ["Flat", "Green", "Dusty"])

# -----------------------------
# PREDICTION
# -----------------------------
if st.sidebar.button("Predict Performance"):

    player = data[data["Player_Name"] == selected_player]

    if player.empty:
        st.error("Player not found")
    else:
        latest_player = player.sort_values("Year").iloc[-1:].copy()

        # Reset condition columns
        for col in data.columns:
            if col.startswith("Weather_"):
                latest_player[col] = 0
            if col.startswith("Pitch_"):
                latest_player[col] = 0

        if f"Weather_{weather_input}" in latest_player.columns:
            latest_player[f"Weather_{weather_input}"] = 1

        if f"Pitch_{pitch_input}" in latest_player.columns:
            latest_player[f"Pitch_{pitch_input}"] = 1

        input_df = latest_player[features]

        predicted_runs = model.predict(input_df)[0]
        current_runs = latest_player["Runs_Scored"].iloc[0]

        # -----------------------------
        # CONFIDENCE ESTIMATION
        # -----------------------------
        tree_predictions = np.array([
            tree.predict(input_df)[0]
            for tree in model.estimators_
        ])

        std_dev = np.std(tree_predictions)

        # -----------------------------
        # CLEAN LAYOUT WITH METRICS
        # -----------------------------
        st.subheader("📊 Performance Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric("Current Runs", int(current_runs))
        col2.metric("Predicted Runs (2026)", int(predicted_runs))
        col3.metric("Prediction Uncertainty (±)", int(std_dev))

        st.markdown(
            f"### 🎯 Expected Runs (2026): **{int(predicted_runs)} ± {int(std_dev)}**"
        )

        # -----------------------------
        # FEATURE IMPORTANCE
        # -----------------------------
        st.subheader("📈 Key Performance Drivers")

        importance = model.feature_importances_

        feat_imp = pd.DataFrame({
            "Feature": features,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        st.bar_chart(feat_imp.set_index("Feature"))

        # -----------------------------
        # PLAYER TREND GRAPH
        # -----------------------------
        st.subheader("📉 Historical Performance Trend")

        trend_df = player.sort_values("Year")[["Year", "Runs_Scored"]]
        st.line_chart(trend_df.set_index("Year"))