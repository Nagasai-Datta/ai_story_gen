import streamlit as st
from transformers import pipeline
import requests
import base64
import time
import concurrent.futures

# Initialize session state for persistent data
if 'generated_story' not in st.session_state:
    st.session_state.generated_story = None
if 'image_data' not in st.session_state:
    st.session_state.image_data = None

# ⚠️ TEMPORARY API KEY - DELETE AFTER TESTING ⚠️
TOGETHER_API_KEY = "3919c23b8c8c46343b2fffc598514bce81365cd3fbd5885f8baf5fc86b0ae5da"

# Initialize the AI model
@st.cache_resource
def load_story_model():
    return pipeline('text-generation', model='gpt2')

# Updated image generation function with working endpoint
def generate_image(prompt, api_key):
    """Generate image using Together AI's Stable Diffusion XL"""
    if not api_key:
        return None
        
    # Use verified working endpoint
    url = "https://api.together.xyz/inference"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Use a more reliable model
    payload = {
        "model": "stabilityai/stable-diffusion-xl-base-1.0",
        "prompt": prompt,
        "height": 512,
        "width": 512,
        "steps": 30,
        "seed": int(time.time()),
        "n": 1  # Number of images to generate
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        # Extract image data from updated response format
        result = response.json()
        if "output" in result and "choices" in result["output"]:
            image_base64 = result["output"]["choices"][0]["image_base64"]
            return base64.b64decode(image_base64)
        else:
            st.error(f"Unexpected API response: {result}")
            return None
            
    except Exception as e:
        st.error(f"⚠️ Image generation failed: {str(e)}")
        if response:
            st.info(f"API response: {response.text[:200]}...")
        return None

# App setup
st.set_page_config(page_title="AI Story Generator", page_icon="📖")
st.title("📖 AI Story Generator")
st.subheader("Turn your ideas into illustrated stories!")

# User input
prompt = st.text_area("Enter your story prompt:", 
                      "A time traveler visits ancient Egypt...", 
                      height=100)

# Settings
col1, col2 = st.columns(2)
with col1:
    length = st.slider("Story Length", 50, 500, 150)
with col2:
    creativity = st.slider("Creativity", 0.5, 1.0, 0.85)

# Generate button
if st.button("✨ Generate Story & Image"):
    if not prompt.strip():
        st.error("Please enter a prompt!")
    else:
        with st.spinner("Creating your story and cover art simultaneously..."):
            try:
                # Run both generation tasks in parallel
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Submit both tasks
                    story_future = executor.submit(lambda: load_story_model()(
                        prompt,
                        max_length=length + len(prompt),
                        temperature=creativity,
                        num_return_sequences=1
                    ))
                    
                    image_future = executor.submit(generate_image, prompt, TOGETHER_API_KEY)
                    
                    # Get results as they complete
                    story_result = story_future.result()
                    image_data = image_future.result()
                
                # Process story
                story = story_result[0]['generated_text'].replace(prompt, "").strip()
                st.session_state.generated_story = story
                
                # Display story
                st.subheader("Your Story:")
                st.write(story)
                
                # Download story option
                st.download_button("📥 Download Story", story, file_name="story.txt")
                
                # Display image if generated
                if image_data:
                    st.session_state.image_data = image_data
                    st.subheader("Story Cover Art:")
                    st.image(image_data, caption="Generated with Stable Diffusion XL")
                    st.download_button(
                        "📥 Download Image", 
                        image_data, 
                        file_name="story_cover.png",
                        mime="image/png"
                    )
                else:
                    st.warning("Image generation failed. Story was still created successfully.")
                
            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")
                st.info("Try a shorter story or different prompt")

# Show previous results if available
if st.session_state.generated_story:
    st.divider()
    st.subheader("Previous Story")
    st.write(st.session_state.generated_story)
    
if st.session_state.image_data:
    st.subheader("Previous Cover Art")
    st.image(st.session_state.image_data)

st.caption("Built with GPT-2 and Stable Diffusion XL")
