# CHAT INTERFACE (GEN AI READY)
# =====================================================
st.markdown("---")
st.markdown('<div class="section-header">💬 AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask about player form, risk, conditions...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Placeholder (OpenAI will replace this later)
    ai_response = "Generative AI module will analyze this once OpenAI is integrated."

    st.session_state.messages.append({"role": "assistant", "content": ai_response})

    st.rerun()