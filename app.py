import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
import re

# Page Configuration
st.set_page_config(page_title="AI Cyberbullying Detector", page_icon="🛡️")

# Load the Model and Tokenizer
@st.cache_resource
def load_assets():
    # Looking for the exact file names you downloaded!
    model = tf.keras.models.load_model('cyberbullying_model (2).keras')
    
    with open('tokenizer.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
        
    return model, tokenizer

try:
    model, tokenizer = load_assets()
    assets_loaded = True
except Exception as e:
    st.error(f"Error loading files: {e}")
    assets_loaded = False

# Text Cleaning Function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# UI Design
st.title("🛡️ Cyberbullying Detection System")
st.markdown("Enter a comment below to check if it contains toxic or bullying content.")

user_input = st.text_area("User Comment:", placeholder="Type a message here...")

if st.button("Analyze Text"):
    if not assets_loaded:
        st.error("Cannot predict because the model/tokenizer failed to load.")
    elif user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Preprocess the input
        cleaned = clean_text(user_input)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=100)
        
        # Predict
        prediction = model.predict(padded)[0][0]
        
        # Display Results
        st.divider()
        st.subheader("Analysis Result:")
        if prediction > 0.5:
            st.error(f"⚠️ **Cyberbullying Detected**")
            st.write(f"Confidence Score: **{prediction*100:.2f}%**")
        else:
            st.success(f"✅ **Normal Message**")
            st.write(f"Confidence Score: **{(1-prediction)*100:.2f}%**")