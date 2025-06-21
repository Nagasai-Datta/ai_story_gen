import requests
import base64
import time

def generate_image(prompt, api_key):
    
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
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  
        
        
        image_base64 = response.json()["output"]["choices"][0]["image_base64"]
        return base64.b64decode(image_base64)
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {str(e)}")
    except KeyError:
        raise Exception("Invalid response format from API")
