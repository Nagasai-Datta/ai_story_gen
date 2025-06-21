import requests
import base64
import time
import streamlit as st

def generate_image(prompt, api_key):
    """Generate image using Together AI's Stable Diffusion XL"""
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
        "seed": int(time.time())  # Random seed
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()  # Raise error for bad status
        
        # Extract image data
        image_base64 = response.json()["output"]["choices"][0]["image_base64"]
        return base64.b64decode(image_base64)
        
    except Exception as e:
        st.error(f"⚠️ Image generation failed: {str(e)}")
        st.info("This might be due to: \n1. Invalid API key \n2. Server overload \n3. NSFW content")
        return None
