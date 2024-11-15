import openai
from config import API_KEY

# Set the API key
openai.api_key = API_KEY

# Test Request
try:
    response = openai.chat.completions.create(
        model="gpt-4o-mini",  # Replace with your desired model, e.g., "gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say this is a test"}
        ]
    )
    print("API Call Successful!")
    print(response['choices'][0]['message']['content'])
except Exception as e:
    print("API Call Failed!")
    print(e)