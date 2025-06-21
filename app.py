import streamlit as st
from transformers import pipeline
import requests
import base64
import time

# ⚠️  INSECURE - REPLACE WITH ENVIRONMENT VARIABLE LATER ⚠️
TOGETHER_API_KEY = "YOUR_API_KEY"  # Will be removed later

# Initialize session state
if 'generated_story' not in st.session_state:
    st.session_state.generated_story = None

# Initialize model
@st.cache_resource
def load_story_model():
    return pipeline('text-generation', model='gpt2')

# Image generation function
def generate_image(prompt, api_key):
    """Generate image using Together AI's Stable Diffusion XL"""
    if not api_key:
        st.error("API key missing! Image generation disabled")
        return None
        
    url = "https://api.together.xyz/v1/images/generations"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "stabilityai/stable-diffusion-xl-base-1.0",
        "prompt": prompt,
        "height": 512,
        "width": 512,
        "steps": 30,
        "seed": int(time.time())
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        image_base64 = response.json()["output"]["choices"][0]["image_base64"]
        return base64.b64decode(image_base64)
        
    except Exception as e:
        st.error(f"⚠️ Image generation failed: {str(e)}")
        st.info("Common causes: 1. Invalid API key 2. Server overload 3. NSFW content")
        return None

# App UI
st.set_page_config(page_title="AI Story Generator", page_icon="📖")
st.title("📖 AI Story Generator")
st.subheader("Turn your ideas into stories!")

prompt = st.text_area("Enter your story prompt:", 
                     "A time traveler visits ancient Egypt...", 
                     height=100)

col1, col2 = st.columns(2)
with col1:
    length = st.slider("Story Length", 50, 250, 100)
with col2:
    creativity = st.slider("Creativity", 0.5, 1.0, 0.8)

# Story generation
if st.button("✨ Generate Story"):
    if not prompt.strip():
        st.error("Please enter a prompt!")
    else:
        with st.spinner("Creating your story..."):
            try:
                generator = load_story_model()
                result = generator(
                    prompt,
                    max_length=length + len(prompt),
                    temperature=creativity,
                    num_return_sequences=1
                )
                story = result[0]['generated_text'].replace(prompt, "").strip()
                st.session_state.generated_story = story
                
                st.subheader("Your Story:")
                st.write(story)
                st.download_button("📥 Download Story", story, file_name="story.txt")
                
            except Exception as e:
                st.error(f"Story generation failed: {str(e)}")

# Display generated story if exists
if st.session_state.generated_story:
    st.divider()
    st.subheader("Generate Story Cover Art")
    
    if st.button("🖼️ Generate Cover Image"):
        with st.spinner("Creating your cover art..."):
            image_data = generate_image(
                prompt=prompt,
                api_key=TOGETHER_API_KEY
            )
            if image_data:
                st.image(image_data, caption="Your Story Cover")
                st.download_button(
                    "📥 Download Image", 
                    image_data, 
                    file_name="story_cover.png",
                    mime="image/png"
                )

st.caption("Built with GPT-2 and Streamlit")
