AI Research Assistant (RAG Pipeline)

OVERVIEW

This is my AI Research Assistant project, built using Streamlit to create a simple web interface. It uses the Gemini API within a Retrieval-Augmented Generation (RAG) pipeline to answer questions by strictly referencing uploaded PDF documents. The RAG components are managed by LangChain and indexed using FAISS.

QUICK START

1. Setup & Dependencies

Make sure you have Python (3.9+) installed. From the project's root directory:

Dependencies: Install all necessary libraries from requirements.txt.

pip install -r requirements.txt


API Key: Create a file named .env in the project root and add your Gemini API key inside it.

GOOGLE_API_KEY="[YOUR_CONFIDENTIAL_KEY]" 


2. Run Application

Execute the main Streamlit script to start the application in your browser.

streamlit run app.py
