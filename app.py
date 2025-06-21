import streamlit as st
from transformers import pipeline
import time

# Initialize the AI model
@st.cache_resource
def load_story_model():
    return pipeline('text-generation', model='gpt2')

# App setup with light theme
st.set_page_config(
    page_title="AI Story Generator", 
    page_icon="📖",
    layout="centered"
)

# Light theme styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stTextArea, .stSlider, .stButton>button {
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
    .stButton>button {
        background-color: #4e73df !important;
        color: white !important;
        font-weight: 500;
        padding: 0.5rem 1rem;
    }
    .stDownloadButton>button {
        background-color: #28a745 !important;
        color: white !important;
    }
    .stSpinner>div {
        border-top-color: #4e73df;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📖 AI Story Generator")
st.subheader("Turn your ideas into captivating stories")

# User input
with st.container():
    st.write("### Your Story Prompt")
    prompt = st.text_area(
        "Enter your starting idea:",
        "A time traveler visits ancient Egypt...", 
        height=120,
        label_visibility="collapsed"
    )

# Settings
st.write("### Story Settings")
col1, col2 = st.columns(2)
with col1:
    length = st.slider("Length (words)", 50, 300, 120)
with col2:
    creativity = st.slider("Creativeness", 0.5, 1.0, 0.8, help="Higher values = more creative/risky")

# Generate button
if st.button("✨ Generate Story", use_container_width=True):
    if not prompt.strip():
        st.error("Please enter a story prompt!")
    else:
        with st.spinner("Crafting your story... This takes about 10-30 seconds"):
            try:
                # Load model
                generator = load_story_model()
                
                # Generate story
                start_time = time.time()
                result = generator(
                    prompt,
                    max_length=length + len(prompt),
                    temperature=creativity,
                    num_return_sequences=1
                )
                gen_time = time.time() - start_time
                
                # Display story
                story = result[0]['generated_text'].replace(prompt, "").strip()
                st.success(f"Story generated in {gen_time:.1f} seconds!")
                
                st.subheader("Your Custom Story")
                st.markdown(f'<div style="background-color: #ffffff; padding: 20px; border-radius: 10px; border-left: 4px solid #4e73df;">{story}</div>', unsafe_allow_html=True)
                
                # Download option
                st.download_button(
                    "📥 Download Story", 
                    story, 
                    file_name="ai_story.txt",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")
                st.info("Try reducing the story length or using a different prompt")

st.caption("Built with GPT-2 and Streamlit • Light theme v1.0")
