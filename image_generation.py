import requests
import base64
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
    
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        image_data = response.json()["output"]["choices"][0]["image_base64"]
        return base64.b64decode(image_data)
    else:
        st.error(f"Image generation failed: {response.text}")
        return None
