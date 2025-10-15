import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from PyPDF2 import PdfReader
import tempfile
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    st.error("GOOGLE_API_KEY not found. Please check your .env file.")
    st.stop()

st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.title("AI Research Assistant")
st.write("Upload a research paper (PDF) and ask questions about it using Gemini + LangChain.")

uploaded_file = st.file_uploader("Upload your research paper (PDF)", type=["pdf"])
user_question = st.text_input("Ask a question about the document:")

if uploaded_file:
    # Check if file is new or chain needs creation
    if "qa_chain" not in st.session_state or st.session_state.get("file_id") != uploaded_file.file_id:
        
        st.session_state.clear()
        st.session_state["file_id"] = uploaded_file.file_id

        with st.spinner("Processing document..."):
            try:
                # 1. Extract Text
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    temp_pdf_path = tmp.name

                text = "".join(page.extract_text() or "" for page in PdfReader(temp_pdf_path).pages)
                os.unlink(temp_pdf_path)

                st.success("Document processed successfully.")

                # 2. Chunk, Embed, and Store
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_text(text)
                
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

                # 3. Setup LLM and QA Chain
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=API_KEY)
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                
                st.session_state["qa_chain"] = ConversationalRetrievalChain.from_llm(
                    llm, vectorstore.as_retriever(), memory=memory
                )
                st.session_state["ready"] = True

            except Exception as e:
                st.error(f"Error during setup: {e}")
                st.session_state["ready"] = False
    
    # Answer Question
    if st.session_state.get("ready", False) and user_question:
        qa_chain = st.session_state["qa_chain"]
        with st.spinner("Generating response..."):
            try:
                response = qa_chain.invoke({"question": user_question})
                st.write("### AI Response")
                st.write(response["answer"])
            except Exception as e:
                st.error(f"Error: {e}")
    elif st.session_state.get("ready", False) and not user_question:
        st.info("The document is ready. Ask your first question!")

else:
    # Clear state when no file is present
    st.session_state.pop("qa_chain", None)
    st.session_state.pop("ready", None)
    st.session_state.pop("file_id", None)
