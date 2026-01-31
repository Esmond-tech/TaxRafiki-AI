import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

# 1. FIXED Google Drive Connection Logic
def get_drive_service():
    # We pull the secrets into a dictionary and manually clean the private key
    # to fix the persistent "ASN.1 unexpected tag" mobile error.
    creds_dict = {
        "type": st.secrets["gcp_service_account"]["type"],
        "project_id": st.secrets["gcp_service_account"]["project_id"],
        "private_key_id": st.secrets["gcp_service_account"]["private_key_id"],
        "private_key": st.secrets["gcp_service_account"]["private_key"].replace("\\n", "\n"),
        "client_email": st.secrets["gcp_service_account"]["client_email"],
        "client_id": st.secrets["gcp_service_account"]["client_id"],
        "auth_uri": st.secrets["gcp_service_account"]["auth_uri"],
        "token_uri": st.secrets["gcp_service_account"]["token_uri"],
        "auth_provider_x509_cert_url": st.secrets["gcp_service_account"]["auth_provider_x509_cert_url"],
        "client_x509_cert_url": st.secrets["gcp_service_account"]["client_x509_cert_url"],
    }
    creds = service_account.Credentials.from_service_account_info(creds_dict)
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
        docs = st.session_state.vector_store.similarity_search(user_question)
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        prompt = PromptTemplate(template="Context:\n{context}?\nQuestion:\n{question}\nAnswer:", input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        
        with st.chat_message("user"): st.write(user_question)
        with st.chat_message("assistant"): st.write(response["output_text"])
