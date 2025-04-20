from dotenv import load_dotenv
import os

class ApiKeyManager:
    def __init__(self):
        load_dotenv()
        self.keys = [
            os.getenv("groq_api_key1"),
            os.getenv("groq_api_key2")
        ]
        self.current_key_index = 0
        
    def get_next_key(self):
        key = self.keys[self.current_key_index]
        self.current_key_index = (self.current_key_index + 1) % len(self.keys)
        return key
    
    def get_current_key(self):
        return self.keys[self.current_key_index]
