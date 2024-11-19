import openai
import os
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken
import json
from datetime import datetime

# Load environment variables from the .env file
load_dotenv()

# Default configurations
DEFAULT_API_KEY = os.environ.get('OPEN_AI_API_KEY') # Fetch the API key from the environment
DEFAULT_MODEL = "gpt-4o-mini" # Default model for generating responses
DEFAULT_BASE_URL = "https://api.openai.com/v1" # Default API endpoint
DEFAULT_TEMPERATURE = 0.7 # Default creativity level for responses
DEFAULT_MAX_TOKENS = 512 # Default maximum tokens for responses
DEFAULT_TOKEN_BUDGET = 4000 # Default token budget for conversation history
DEFAULT_HISTORY_FILE = f"conversation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"  # Unique history file

class ConversationManager:
    """
    A class to manage interactions with the OpenAI API for a chatbot with token management and persistent conversation history.
    """
    def __init__(self, api_key=None, base_url=None, model=None, history_file=None, temperature=None, max_tokens=None, token_budget=None):
        """
        Initializes the ConversationManager with customizable parameters for API configuration,
        chatbot persona, and token management using Default fall back and Ternary operators.

        - Loads conversation history from the specified file or starts with a default system message.
        - Configures the OpenAI API client.

        :param api_key: The API key for OpenAI. Defaults to the key from the environment.
        :param base_url: The base URL for the API. Defaults to the OpenAI endpoint.
        :param model: The model to use for generating responses. Defaults to DEFAULT_MODEL.
        :param history_file: The file name to store/load conversation history. Defaults to a unique file name.
        :param temperature: Creativity level of generated responses. Defaults to DEFAULT_TEMPERATURE
        :param max_tokens: Maximum lenght of generated responses. Defaults to the DEFAULT_MAX_TOKENS.
        :param token_budget: Total token budget for the conversation. Defaults to DEFAULT_TOKEN_BUDGET.
        """

        if api_key is None:
            self.api_key=DEFAULT_API_KEY
        if base_url is None:
            self.base_url=DEFAULT_BASE_URL
        
        self.model = model if model else DEFAULT_MODEL
        self.temperature = temperature if temperature else DEFAULT_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens else DEFAULT_MAX_TOKENS
        self.token_budget = token_budget if token_budget else DEFAULT_TOKEN_BUDGET
        self.history_file = history_file if history_file else DEFAULT_HISTORY_FILE

        # Predefined system messages or different personas
        self.system_messages = {
            "sassy_assistant":"You are a sassy assistant who is fed up with answering questions.",
            "angry_assistant":"You are an angry assistant that likes yelling in all caps.",
            "thoughtful_assistant":"You are a thoughtful assistant, always ready to dig deeper. You ask clarifying questions to ensure you understood and approach problems with a step-by-step methodology.",
            "custom":"Enter your custom message here."
        }

        # Set the default persona and load the conversation history
        self.system_message = self.system_messages["sassy_assistant"] # Default persona
        self.conversation_history = [{"role":"system", "content":self.system_message}]
        self.load_conversation_history()

        # Configure the OpenAI API client
        self.client = OpenAI(api_key = self.api_key, base_url = self.base_url)

    # Utility Methods
    def load_conversation_history(self):
        """
        Loads the conversation history fro the specified file.

        - If the file does not exist or is invalid, initializes with the default system message.
        """
        try:
            with open(self.history_file, "r") as file:
                self.conversation_history = json.load(file)
        except FileNotFoundError:
            print(f"No existing history file found. Starting fresh with system message.")
            self.conversation_history = [{"role":"system", "content": self.system_message}]
        except json.JSONDecodeError:
            print("Error reading the conversation history file. Starting fresh with system message.")
            self.conversation_history = [{"role":"system", "content": self.system_message}]
        except Exception as e:
            print(f"Unexpected error loading conversation history: {e}")

    def save_conversation_history(self):
        """
        Saves the current conversation history to the specified file in JSON format.
        """
        try:
            with open(self.history_file, "w") as file:
                json.dump(self.conversation_history, file, indent=4)
        except IOError as e:
            print(f"An I/O error occured while saving the conversation history: {e}")
        except Exception as e:
            print(f"An unexpected error occured while saving the conversation history: {e}")

    def reset_conversation_history(self):
        """
        Resets the conversation history to the defaults system message and saves it to the file.
        """
        self.conversation_history = [{"role":"system", "content": self.system_message}]
        self.save_conversation_history()
        print("Conversation history has been reset.")     

    def enforce_token_budget(self):
        """
        Trims the conversation history to fit within the token budget by removing the oldest messages.

        - Ensures the system message is always preserved.
        """
        while self.total_tokens_used() > self.token_budget:
            if len(self.conversation_history) > 1:
                self.conversation_history.pop(1)  # Remove oldest user/assistant message
            else:
                break

    def total_tokens_used(self):
        """
        Calculates the total number of tokens in the conversation history.

        :return: The total number of tokens used in the conversation history.
        """
        return sum(self.count_tokens(message["content"]) for message in self.conversation_history)

    def count_tokens(self, text):
        """
        Counts the number of tokens in a given piece of text.

        :param text: The text to encode and count tokens.
        :return: The number of tokens in the text.
        """
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    
    # Persona Management
    def set_persona(self, persona):
        """
        Changes the chatbot's persona by updating the system message.

        :param persona: The persona to set. Must be one of the predefined personas.
        :raises ValueError: If the persona is not recognized.
        """        
        if persona in self.system_messages:
            self.system_message = self.system_messages[persona]
            self.update_system_message_in_history()
        else:
            raise ValueError(f"Unknown persona: {persona}. Available personas are: {list(self.system_messages.keys())}")

    def set_custom_system_message(self, custom_message):
        """
        Sets a custom system message for the chatbot.

        :param custom_message: The custom message to set.
        :raises ValueError: If the custom message is empty.
        """
        if not custom_message.strip():
            raise ValueError("Custom message cannot be empty.")
        self.system_messages["custom"] = custom_message
        self.set_persona("custom")

    def update_system_message_in_history(self):
        """
        Updates the system message in the conversation history to match the current persona.
        """
        if self.conversation_history and self.conversation_history[0]["role"] == "system":
            self.conversation_history[0]["content"] = self.system_message
        else:
            self.conversation_history.insert(0, {"role":"system", "content": self.system_message})

# Chat Completion
    def chat_completion(self, prompt, temperature=None, max_tokens=None, model=None):
        """
        Sends a prompt to the OpenAI API and retrieves the chatbot's response.

        :param prompt: The user's input message to the chatbot.
        :param temperature: (Optional) Creativity level for the response. Overrides the the default if provided.
        :param max_tokens: (Optional) Maximum tokens for the response. Overrides the default if provided.
        :param model: (Optional) The model to use for this request. Overrides the default if provided.
        :return: the chatbot's response as a string.
        """
        try:
            # Use dynamic of default parameters using the None check
            temperature = temperature if temperature is not None else self.temperature
            max_tokens = max_tokens if max_tokens is not None else self.max_tokens

            # Append the user's prompt to the conversation history
            self.conversation_history.append({"role":"user", "content":prompt})

            # Ensures the conversation history respects the token budget
            self.enforce_token_budget()

            # Call the OpenAI API with the updated conversation history
            response = self.client.chat.completions.create(
                model = self.model,
                messages = self.conversation_history,
                temperature = temperature,
                max_tokens = max_tokens
            )

            # Extract the AI's response and append it to the conversation history
            ai_response = response.choices[0].message.content
            self.conversation_history.append({"role":"assistant", "content":ai_response})
            # Save the conversation history after the chat completion
            self.save_conversation_history()

            return ai_response
        except openai.error.OpenAIerror as e:
            print(f"OpenAI API error: {e}")
            return "I am sorry, but I can't process that request right now."
        except Exception as e:
            print(f"Unexpected error: {e}")
            return "An error occured. Please try again."
                    
# Test script
test_prompts = [
    "What is the capital of France?",
    "Tell me a joke.",
    "Who wrote 'Pride and Prejudice'?",
    "What is 2 + 2?",
    "Can you explain why the answer is correct?",
    {"persona": "sassy_assistant", "prompt": "Why do you always sound so grumpy?"},
    {"persona": "angry_assistant", "prompt": "Why are you yelling at me?"},
    {"persona": "thoughtful_assistant", "prompt": "Can you help me write a step-by-step plan to solve a problem?"},
    {"persona": "custom", "prompt": "You are a Bahá'í with profound knowledge of the Faith. Can you explain the principle of progressive revelation?"},
    "",
    "Tell me something so long it breaks the token budget" * 500
]

def test_chatbot(manager, prompts):
    for i, prompt in enumerate(prompts):
        if isinstance(prompt, dict):
            manager.set_persona(prompt["persona"])
            prompt_text = prompt["prompt"]
        else:
            prompt_text = prompt
        
        print(f"Test {i + 1}: {prompt_text}")
        response = manager.chat_completion(prompt_text)
        print(f"Response: {response}\n")
        print("-" * 50)

# Initialize the chatbot manager
conv_manager = ConversationManager()

# Run tests
test_chatbot(conv_manager, test_prompts)