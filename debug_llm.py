
import os
import requests
import json
from config.settings import LLM_API_KEY, LLM_MODEL, LLM_ENDPOINT

def test_native():
    print("\n--- Testing NATIVE Gemini API ---")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{LLM_MODEL}:generateContent?key={LLM_API_KEY}"
    payload = {
        "contents": [{
            "parts": [{"text": "Hello"}]
        }]
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Success! Native API works.")
        else:
            print(f"Failed. Response: {response.text}")
            
    except Exception as e:
        print(f"Native Test Error: {e}")

def test_openai_compat():
    print("\n--- Testing OPENAI-COMPATIBLE API ---")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}"
    }

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "max_tokens": 10
    }

    try:
        response = requests.post(LLM_ENDPOINT, headers=headers, json=payload, timeout=30)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")

    except Exception as e:
        print(f"OpenAI Compat Test Error: {e}")

if __name__ == "__main__":
    print(f"Testing Model: {LLM_MODEL}")
    test_native()
    test_openai_compat()
