import streamlit as st
import os

# 1. Securely load the API key from Streamlit Secrets
if "LLM_API_KEY" in st.secrets:
    os.environ["LLM_API_KEY"] = st.secrets["LLM_API_KEY"]
else:
    st.error("API Key not found! Please add LLM_API_KEY to Streamlit Secrets.")

# 2. Import your backend functions directly from app.py
# Make sure app.py is in the same folder!
try:
    from app import extract_data_with_llm, analyze_sentiment_with_llm
except ImportError as e:
    st.error(f"Error importing from app.py: {e}")
    st.stop()

# 3. Streamlit UI
st.set_page_config(page_title="Flash Insights AI", page_icon="⚡")
st.title("⚡ Flash Insights AI")
st.write("Welcome, judges! Select a task and enter your text below.")

# Let the user choose what they want to do
task = st.selectbox("Choose a task:", ["Analyze Sentiment", "Extract Data"])
user_text = st.text_area("Enter your text here:", height=150)

if st.button("Generate Insights"):
    if user_text.strip():
        with st.spinner("Analyzing with Gemini AI..."):
            try:
                # Call your functions directly!
                if task == "Analyze Sentiment":
                    result = analyze_sentiment_with_llm(user_text)
                else:
                    result = extract_data_with_llm(user_text)
                
                st.success("Analysis Complete!")
                st.json(result) # Display the JSON result nicely
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter some text first!")