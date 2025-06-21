import streamlit as st
from transformers import pipeline
import time

# Initialize the AI model
@st.cache_resource
def load_story_model():
    return pipeline('text-generation', model='gpt2')

# App setup
st.set_page_config(page_title="AI Story Generator", page_icon="📖")
st.title("📖 AI Story Generator")
st.subheader("Turn your ideas into stories!")

# User input
prompt = st.text_area("Enter your story prompt:", 
                      "A time traveler visits ancient Egypt...", 
                      height=100)

# Settings
col1, col2 = st.columns(2)
with col1:
    length = st.slider("Story Length", 50, 250, 100)
with col2:
    creativity = st.slider("Creativity", 0.5, 1.0, 0.8)

# Generate button
if st.button("✨ Generate Story"):
    if not prompt.strip():
        st.error("Please enter a prompt!")
    else:
        with st.spinner("Creating your story..."):
            try:
                # Load model
                generator = load_story_model()
                
                # Generate story
                result = generator(
                    prompt,
                    max_length=length + len(prompt),
                    temperature=creativity,
                    num_return_sequences=1
                )
                
                # Display story
                story = result[0]['generated_text'].replace(prompt, "").strip()
                st.subheader("Your Story:")
                st.write(story)
                
                # Download option
                st.download_button("📥 Download Story", story, file_name="story.txt")
                
            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")
                st.info("Try a shorter story or different prompt")

st.caption("Built with GPT-2 and Streamlit")