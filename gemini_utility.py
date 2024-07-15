import os
import json
# from PIL import Image

import google.generativeai as genai

# Get the Working Directory 
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE_PATH = f"{WORKING_DIR}/config.json"

config_data = json.load(open(CONFIG_FILE_PATH))
# Loading the Gemini API Key
GOOGLE_API_KEY = config_data['GOOGLE_API_KEY']
# Configuring the google.generativeai with the Gemini API Key
genai.configure(api_key=GOOGLE_API_KEY)

# Function Definition for Gemini-Pro model
def load_gemini_pro_model():
    # gemini_pro_model = genai.GenerativeModel("gemini-pro")
    gemini_pro_model = genai.GenerativeModel("gemini-1.0-pro")
    return gemini_pro_model

# Function Definition for Gemini-Pro-Vision model
def gemini_pro_vision_response(prompt, image):
    # gemini_pro_vision_model = genai.GenerativeModel("gemini-pro-vision-latest")
    gemini_pro_vision_model = genai.GenerativeModel("gemini-pro-vision")
    response = gemini_pro_vision_model.generate_content([prompt, image])
    text_response = response.text
    return text_response

# Function definition for Text-Embedding Model
def text_embedding_model(input_text):
    embedding_model = "models/embedding-001"
    embedding = genai.embed_content(
        model=embedding_model,
        content=input_text,
        task_type='retrieval_document'
    )
    embedding_list = embedding['embedding']
    return embedding_list

def ask_gemini_response(user_prompt):
    gemini_pro_model = genai.GenerativeModel("gemini-1.0-pro")
    response = gemini_pro_model.generate_content(user_prompt)
    text_result = response.text
    return text_result
    

'''
TESTING THE GEMINI PARTS
'''
# IMAGE CAPTION GENERATION
# image = Image.open("test_image.png")
# prompt = "write a short caption for this given image"
# output = gemini_pro_vision_response(prompt, image)
# print(output)

# TEXT EMBEDDING
# input_text = "betty bought some butter. the butter was bitter"
# print(text_embedding_model(input_text))