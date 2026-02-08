# dashboard.py (Full Updated Version)

import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import time

# ============================================
# APP CONFIGURATION
# ============================================

st.set_page_config(
    page_title="MindCompass | Multimodal AI",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

API_URL = "http://localhost:8000/predict"
CHAT_URL = "http://localhost:8000/chat"

# ============================================
# MODERN DESIGN SYSTEM (CSS)
# ============================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Plus+Jakarta+Sans:wght@300;400;600&display=swap');

    :root {
        --glass-bg: rgba(255, 255, 255, 0.7);
        --glass-border: rgba(255, 255, 255, 0.4);
        --accent-primary: #6366f1;
        --accent-secondary: #a855f7;
        --text-main: #1e293b;
    }

    .stApp {
        background: radial-gradient(circle at top right, #e0e7ff, transparent),
                    radial-gradient(circle at bottom left, #f3e8ff, transparent),
                    #f8fafc;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Compact Glassmorphic Navbar */
    .nav-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.8rem 5%;
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(12px);
        border-bottom: 1px solid var(--glass-border);
        position: sticky;
        top: 0;
        z-index: 1000;
        margin-bottom: 1.5rem;
    }

    .nav-logo {
        font-family: 'Outfit', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4f46e5, #9333ea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Hero Section - Explicitly Centered */
    .hero-container {
        text-align: center !important;
        padding: 1.5rem 10% 2.5rem 10%;
        width: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    /* Global Text Styles - High Contrast */
    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: #1e293b !important;
    }

    .hero-title {
        text-align: center !important;
        font-family: 'Outfit', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        color: #1e293b !important;
        line-height: 1.1;
        margin-bottom: 0.8rem;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #475569 !important; /* Slightly softer but still high contrast */
        max-width: 850px;
        margin: 0 auto !important;
        text-align: center !important;
        line-height: 1.6;
    }

    /* Navbar items color fix */
    .nav-container span {
        color: #64748b !important;
    }
    
    .nav-logo {
        background: linear-gradient(90deg, #4f46e5, #9333ea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Content Alignment - Strictly Left Aligned */
    [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] {
        justify-content: flex-start !important;
    }
    
    .stRadio > div {
        justify-content: flex-start !important;
        gap: 2rem !important;
        margin: 0.5rem 0 1.5rem 0 !important;
    }

    h4 {
        text-align: left !important;
        width: 100% !important;
        margin-bottom: 0.5rem !important;
    }

    /* Glassmorphic Columns */
    [data-testid="column"] {
        background: var(--glass-bg);
        backdrop-filter: blur(16px);
        border-radius: 24px;
        border: 1px solid var(--glass-border);
        padding: 2rem !important;
        min-height: 480px !important;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
        gap: 0.8rem;
    }

    /* Input boxes - Force Black Background & White Text */
    .stTextArea textarea, 
    [data-testid="stFileUploaderDropzone"] {
        height: 250px !important;
        min-height: 250px !important;
        border: 1px dashed #cbd5e1 !important;
        border-radius: 16px !important;
        background-color: #000000 !important;
        color: #ffffff !important;
    }

    /* Target all nested text in file uploader */
    [data-testid="stFileUploaderDropzone"] * {
        color: #ffffff !important;
    }

    /* Fix Text Area Placeholder and Content */
    .stTextArea textarea {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    
    .stTextArea textarea::placeholder {
        color: rgba(255, 255, 255, 0.5) !important;
        -webkit-text-fill-color: rgba(255, 255, 255, 0.5) !important;
    }

    /* Predict Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        color: white !important;
        border: none;
        padding: 1rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 14px;
        width: 100%;
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin-top: 1rem;
    }

    /* Guardian Chat Input */
    [data-testid="stChatInput"] {
        background-color: transparent !important;
        border: none !important;
    }
    
    [data-testid="stChatInput"] textarea {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important; 
        background-color: #334155 !important;
    }

    [data-testid="stChatMessage"] {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(8px) !important;
        border-radius: 18px !important;
        border: 1px solid var(--glass-border) !important;
        margin-bottom: 1rem !important;
    }

    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        text-align: center;
        padding: 1rem;
        background: rgba(248, 250, 252, 0.95);
        backdrop-filter: blur(12px);
        color: #94a3b8;
        font-size: 0.85rem;
        z-index: 1000;
        border-top: 1px solid #e2e8f0;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .animate-in { animation: fadeIn 0.8s ease-out forwards; }

    /* FORCE PERFECT CIRCULAR BUBBLE TO BOTTOM RIGHT */
    button[kind="secondary"][key="guardian_fab_v7"],
    button[key="guardian_fab_v7"] {
        position: fixed !important;
        bottom: 30px !important;
        right: 30px !important;
        width: 85px !important;
        height: 85px !important;
        border-radius: 50% !important;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
        color: white !important;
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.6) !important;
        z-index: 999999 !important;
        font-size: 38px !important;
        padding: 0 !important;
    }

    button[key="guardian_fab_v7"]:hover {
        transform: scale(1.1) rotate(15deg) !important;
        box-shadow: 0 20px 60px rgba(99, 102, 241, 0.8) !important;
    }

    /* Floating Chat Window */
    .floating-guardian-window {
        position: fixed;
        bottom: 125px;
        right: 30px;
        width: 420px;
        background: #0f172a;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 30px;
        box-shadow: 0 40px 100px rgba(0,0,0,0.9);
        z-index: 999998;
        display: flex;
        flex-direction: column;
        overflow: hidden;
        animation: chatSlideUpFinal 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }

    @keyframes chatSlideUpFinal {
        from { transform: translateY(50px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }

    .guardian-header-final {
        padding: 26px 30px;
        background: linear-gradient(90deg, rgba(99, 102, 241, 0.25), rgba(168, 85, 247, 0.25));
        border-bottom: 2px solid rgba(255, 255, 255, 0.1);
        display: flex;
        align-items: center;
        gap: 18px;
    }

    .guardian-title-text {
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 24px !important;
    }

    .guardian-chat-history {
        flex: 1;
        overflow-y: auto;
        padding: 30px;
        min-height: 380px;
        max-height: 480px;
        display: flex;
        flex-direction: column;
        background: #0f172a !important;
    }

    .msg-unit {
        padding: 16px 22px;
        border-radius: 22px;
        margin-bottom: 20px;
        font-size: 15.5px;
        line-height: 1.6;
        max-width: 85%;
        color: white !important;
    }
    
    .msg-unit-user {
        background: #4f46e5;
        align-self: flex-end;
    }
    
    .msg-unit-bot {
        background: #1e293b;
        align-self: flex-start;
    }

    .guardian-footer-input {
        padding: 22px 30px;
        background: #020617;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# UI COMPONENTS
# ============================================

def navbar():
    st.markdown("""
        <div class="nav-container">
            <div class="nav-logo">MindCompass AI</div>
            <div style="display: flex; gap: 2rem; align-items: center;">
                <span style="color: #64748b; font-weight: 500; cursor: pointer;">Dashboard</span>
                <span style="color: #64748b; font-weight: 500; cursor: pointer;">Resources</span>
                <div style="width: 40px; height: 40px; background: #e2e8f0; border-radius: 50%; display: grid; place-items: center;">👤</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Session State Initialization
if "last_data" not in st.session_state:
    st.session_state.last_data = {}
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_guardian_mode" not in st.session_state:
    st.session_state.show_guardian_mode = False

# Navbar
navbar()

# Hero
st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">MindCompass: Multimodal AI</h1>
        <p class="hero-subtitle">Unlock deeper emotional insights through fused analysis of text, voice, and facial expressions. Powered by advanced AI for a premium mental wellness experience.</p>
    </div>
""", unsafe_allow_html=True)

# Input Modes
col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.subheader("Text Input")
    text_mode = st.radio("", ["Free Writing", "Guided Prompt"], horizontal=True, key="text_mode")
    if text_mode == "Free Writing":
        text = st.text_area("", placeholder="How was your day? Write freely...", height=250)
    else:
        text = st.text_area("", placeholder="Guided: Describe a recent challenge and how it made you feel...", height=250)

with col2:
    st.subheader("Audio Input")
    audio_mode = st.radio("", ["Upload Audio", "Live Record"], horizontal=True, key="audio_mode")
    if audio_mode == "Upload Audio":
        audio = st.file_uploader("Drag and drop file here\nLimit 200MB per file • WAV", type=["wav"])
    else:
        st.info("Live Recording Coming Soon!")

with col3:
    st.subheader("Video Input")
    video_mode = st.radio("", ["Upload Video", "Live Capture"], horizontal=True, key="video_mode")
    if video_mode == "Upload Video":
        video = st.file_uploader("Drag and drop file here\nLimit 200MB per file • MP4", type=["mp4"])
    else:
        st.info("Live Capture Coming Soon!")

# Predict Button
if st.button("✨ Generate Emotional Insights"):
    with st.spinner("Analyzing your mind's compass..."):
        files = {}
        if audio:
            files['audio'] = (audio.name, audio.getvalue(), audio.type)
        if video:
            files['video'] = (video.name, video.getvalue(), video.type)
        
        response = requests.post(API_URL, data={"text": text}, files=files)
        
        if response.ok:
            data = response.json()
            st.session_state.last_data = data
            
            col_insight1, col_insight2, col_insight3 = st.columns(3)
            
            with col_insight1:
                st.subheader("Individual Modalities")
                if 'text' in data:
                    st.write(f"Text: {data['text']['emotion'].capitalize()} ({data['text']['confidence']:.2%})")
                if 'audio' in data:
                    st.write(f"Audio: {data['audio']['emotion'].capitalize()} ({data['audio']['confidence']:.2%})")
                if 'video' in data:
                    st.write(f"Video: {data['video']['emotion'].capitalize()} ({data['video']['confidence']:.2%})")
            
            with col_insight2:
                st.subheader("Fusion Prediction")
                fusion = data['fusion']
                st.write(f"**Emotion:** {fusion['emotion'].capitalize()}")
                st.write(f"**Confidence:** {fusion['confidence']:.2%}")
                
                # Display RAG description under emotion
                if 'response' in data and 'answer' in data['response']:
                    st.subheader("Mental Health Insights")
                    st.write(data['response']['answer'])
            
            with col_insight3:
                st.subheader("Modality Weights")
                if 'modality_weights' in fusion:
                    df_weights = pd.DataFrame(list(fusion['modality_weights'].items()), columns=['Modality', 'Weight'])
                    fig = px.pie(df_weights, values='Weight', names='Modality', hole=0.3)
                    st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Probability Distribution")
                df_probs = pd.DataFrame(list(fusion['probs'].items()), columns=['Emotion', 'Probability'])
                fig2 = px.bar(df_probs, x='Emotion', y='Probability', color='Emotion')
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.error("The compass lost its way. Please ensure the backend server is active.")

# Floating Guardian Chat
if st.button("🛡️", key="guardian_fab_v7"):
    st.session_state.show_guardian_mode = not st.session_state.show_guardian_mode
    st.rerun()

if st.session_state.show_guardian_mode:
    msgs_html = ""
    for m in st.session_state.messages:
        tipo = "msg-unit-user" if m["role"] == "user" else "msg-unit-bot"
        msgs_html += f'<div class="msg-unit {tipo}">{m["content"]}</div>'

    st.markdown(f"""
        <div class="floating-guardian-window">
            <div class="guardian-header-final">
                <span style="font-size: 30px;">💠</span>
                <div class="guardian-title-text">Guardian AI</div>
            </div>
            <div class="guardian-chat-history">
                {msgs_html}
            </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="guardian-footer-input">', unsafe_allow_html=True)
        
        # Use a key that changes when we want to reset
        if "chat_input_key" not in st.session_state:
            st.session_state.chat_input_key = 0
        
        prompt = st.text_input(
            "Speak with your Guardian...",
            key=f"guardian_input_{st.session_state.chat_input_key}",
            label_visibility="collapsed"
        )
        
        if prompt:
            # Prevent duplicate processing
            if "last_processed_prompt" not in st.session_state or st.session_state.last_processed_prompt != prompt:
                st.session_state.last_processed_prompt = prompt
                
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Get emotion context
                last_emotion = st.session_state.get('last_data', {}).get('fusion', {}).get('emotion', '')
                
                # Call backend
                try:
                    chat_response = requests.post(
                        CHAT_URL,
                        json={"message": prompt, "emotion": last_emotion}
                    )
                    chat_response.raise_for_status()  # raise if not 200
                    bot_text = chat_response.json()['response']
                    st.session_state.messages.append({"role": "assistant", "content": bot_text})
                except Exception as e:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"Sorry, there was an error: {str(e)}"}
                    )
                
                # Force new input widget on next run (clears the field)
                st.session_state.chat_input_key += 1
            
            # Always rerun to show new messages and clear input
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
     <div class="footer">
         💠 MindCompass AI • DeepMind Premium Experience • © 2026
     </div>
""", unsafe_allow_html=True)