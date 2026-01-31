import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# UPDATED FOR 2026: Use the modern combine_documents path
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

# 1. FIXED Google Drive Connection Logic
def get_drive_service():
    # We pull secrets into a dictionary and fix the mobile newline bug
    creds_info = dict(st.secrets["gcp_service_account"])
    creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")
    
    creds = service_account.Credentials.from_service_account_info(creds_info)
    return build('drive', 'v3', credentials=creds)

def download_pdfs_from_drive(folder_id):
    service = get_drive_service()
    query = f"'{folder_id}' in parents and mimeType='application/pdf'"
    results = service.files().list(q=query).execute()
    items = results.get('files', [])
    
    pdf_texts = ""
    for item in items:
        request = service.files().get_media(fileId=item['id'])
        file_io = io.BytesIO()
        downloader = MediaIoBaseDownload(file_io, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        
        file_io.seek(0)
        reader = PdfReader(file_io)
        for page in reader.pages:
            pdf_texts += page.extract_text()
    return pdf_texts

# 2. RAG Logic
def get_vectorstore(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_texts(chunks, embedding=embeddings)

# --- UI LOGIC ---
st.set_page_config(page_title="TaxRafiki AI", page_icon="🇰🇪")
st.title("🇰🇪 TaxRafiki: Auto-Sync KRA Guide")

# REPLACE THIS ID with your actual Google Drive Folder ID
FOLDER_ID = "1wZ2x3y4v5u6t7s8r9q0p_ExampleID" 

with st.sidebar:
    st.info("Files are synced automatically from Google Drive.")
    if st.button("🔄 Sync Latest Laws"):
        with st.spinner("Fetching KRA documents from Drive..."):
            try:
                raw_text = download_pdfs_from_drive(FOLDER_ID)
                if raw_text:
                    st.session_state.vector_store = get_vectorstore(raw_text)
                    st.success(f"Successfully synced documents!")
                else:
                    st.warning("No PDFs found in the folder.")
            except Exception as e:
                st.error(f"Sync Error: {e}")

# Chat Interface
user_question = st.chat_input("Ask a tax question...")

if user_question:
    if "vector_store" not in st.session_state:
        st.error("Please click 'Sync Latest Laws' in the sidebar first!")
    else:
        # Modern 2026 Chain Pattern
        docs = st.session_state.vector_store.similarity_search(user_question)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
        prompt = ChatPromptTemplate.from_template("Answer based only on context: {context}\nQuestion: {input}")
        
        chain = create_stuff_documents_chain(llm, prompt)
        response = chain.invoke({"context": docs, "input": user_question})
        
        with st.chat_message("user"): st.write(user_question)
        with st.chat_message("assistant"): st.write(response)
