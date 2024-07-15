import os

from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu

from gemini_utility import (
    load_gemini_pro_model, gemini_pro_vision_response, text_embedding_model, ask_gemini_response
)

# WORKING_DIR = "/Users/rohitsriram/Documents/ML-Develop/projects/Gemini/gemini_chatbot/"
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
# print(WORKING_DIR)

# Setting up Streamlit Page Config
st.set_page_config(
    page_title='Gemini AI',
    page_icon='⚜️',
    layout='centered'
)

with st.sidebar:
    selected = option_menu(
        "Gemini AI",
        ["ChatBot", "Image Captioning", "Embed Text", "Ask Anything"],
        menu_icon='robot', 
        icons=['chat-dots-fill', 'image-fill', 'textarea-t', 'patch-question-fill'],
        default_index=0
    )

def translate_role_for_streamlit(user_role):
    if user_role == 'model':
        return 'assistant'
    else:
        return user_role

# CHATBOT FUNCTIONALITY 
if selected == "ChatBot":
    model = load_gemini_pro_model()
    # If chat session in Streamlit is not already running 
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[]) 
            # the empty list contains previous conversation information 
            # it is initially empty because the session is started for the first time 
        
    # Stremlit Page Title
    st.title("🤖 ChatBot")
    
    # Display Chat History
    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text)
            
    # Input field for user (User's prompt to ask the model a question)
    user_prompt = st.chat_input("Ask Gemini-Pro anything")
    if user_prompt:
        st.chat_message('user').markdown(user_prompt)
        # Storing the response from Gemini-Pro Model in a variable
        gemini_response = st.session_state.chat_session.send_message(user_prompt)
        
        # Display the Gemini-pro model response
        with st.chat_message('assistant'):
            st.markdown(gemini_response.text)

# IMAGE CAPTION GENERATOR FUNCTIONALITY             
if selected == "Image Captioning":
    # if 'chat_session' not in st.session_state:
    #     st.session_state.chat_session = model.start_chat(history=[])    
    st.title("✍🏽 Image Caption Generation")
    upload_img = st.file_uploader("Upload an Image!", type=['jpg', 'jpeg', 'png'])
   
    if st.button("Generate Caption"):
        image = Image.open(upload_img)
        col1, col2 = st.columns(2)
        with col1:
            resized_img = image.resize((750, 500))
            st.image(resized_img)
            
        default_prompt = "write a short caption for this given image"
        caption = gemini_pro_vision_response(default_prompt, image)
        
        with col2:
            st.info(caption)
            
# TEXT EMBEDDING FUNCTIONALITY
if selected == "Embed Text":
    st.title("🔤 Text Embedding")
    input_text = st.text_area(label="", placeholder="Enter text to embed")
    
    if st.button("Get Embeddings"):
        response = text_embedding_model(input_text)
        st.markdown(response)
        

if selected == "Ask Anything":
    st.title("⁉️ Ask me anything!")
    user_prompt = st.text_area(label='', placeholder="Ask me a question!")
    if st.button("Get Response"):
        response = ask_gemini_response(user_prompt)
        st.markdown(response)