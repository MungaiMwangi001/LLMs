import os
from dotenv import load_dotenv
from groq import Groq
from IPython.display import display, Markdown

# Load variables from .env
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable not set")   

# Initialize client
client = Groq(api_key=api_key)
models = client.models.list()
print("Available models:", models)

# Create a chat completion
chat_completion = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are a professional Data Scientist."},        
        {"role": "user", "content": "Can you explain how neural networks work?"}
    ],
    model="openai/gpt-oss-120b",
    max_tokens=100,
    temperature=0.5
)

# Extract and display
response_text = chat_completion.choices[0].message.content
display(Markdown(response_text))
