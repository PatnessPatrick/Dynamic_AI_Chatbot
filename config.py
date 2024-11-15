from dotenv import load_dotenv
import os

# Load Environment Variables from the .env file
load_dotenv(dotenv_path=".env")

# Access Variables
API_KEY = os.getenv('API_KEY')
