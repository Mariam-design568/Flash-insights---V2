import streamlit as st
import requests

# 1. Set up the page design
st.set_page_config(page_title="Flash Insights", page_icon="⚡")
st.title("⚡ Flash Insights AI")
st.write("Welcome! Enter your query below to test the AI.")

# 2. IMPORTANT: Put your Render URL here later!
# For now, we leave it as localhost just to test it
BACKEND_URL = "https://your-app-name.onrender.com/your-endpoint" 

# 3. Create the user interface (Text box and Button)
user_query = st.text_input("What would you like to analyze?")

if st.button("Generate Insights"):
    if user_query:
        with st.spinner("Connecting to backend to analyze..."):
            try:
                # 4. Send the data to your app.py backend
                payload = {"query": user_query} 
                response = requests.post(BACKEND_URL, json=payload)
                
                # 5. Display the result on the screen
                if response.status_code == 200:
                    st.success("Analysis Complete!")
                    st.json(response.json()) # Shows the result nicely
                else:
                    st.error(f"Backend Error: {response.status_code}")
                    
            except Exception as e:
                st.error(f"Could not connect to the backend: {e}")
    else:
        st.warning("Please type something first!")