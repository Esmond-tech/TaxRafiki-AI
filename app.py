import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
# STABLE IMPORT FOR 2026
from langchain_core.prompts import PromptTemplate
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

# 1. Drive Connection Logic
def get_drive_service():
    try:
        info = dict(st.secrets["gcp_service_account"])
        info["private_key"] = info["private_key"].replace("\\n", "\n")
        creds = service_account.Credentials.from_service_account_info(info)
        return build('drive', 'v3', credentials=creds)
    except Exception as e:
        st.error(f"Secret Configuration Error: {e}")
        return None

def download_pdfs(folder_id):
    service = get_drive_service()
    if not service: return ""
    query = f"'{folder_id}' in parents and mimeType='application/pdf'"
    results = service.files().list(q=query).execute()
    pdf_texts = ""
    for item in results.get('files', []):
        request = service.files().get_media(fileId=item['id'])
        file_io = io.BytesIO()
        downloader = MediaIoBaseDownload(file_io, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        file_io.seek(0)
        reader = PdfReader(file_io)
        for page in reader.pages:
            pdf_texts += page.extract_text() or ""
    return pdf_texts

# --- UI LOGIC ---
st.set_page_config(page_title="TaxRafiki AI", page_icon="🇰🇪")
st.title("🇰🇪 TaxRafiki: KRA Guide Sync")

FOLDER_ID = "11gCstGrg63yaIH2DTfsEfq6zE11bRzQp"

with st.sidebar:
    if st.button("🔄 Sync KRA Laws"):
        with st.spinner("Processing documents..."):
            raw_text = download_pdfs(FOLDER_ID)
            if raw_text:
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=st.secrets["GOOGLE_API_KEY"])
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = text_splitter.split_text(raw_text)
                st.session_state.vector_store = FAISS.from_texts(chunks, embedding=embeddings)
                st.success("Laws Synced Successfully!")

user_question = st.chat_input("Ask about Kenyan Tax Law...")
if user_question:
    if "vector_store" not in st.session_state:
        st.warning("Please Sync Laws first.")
    else:
        docs = st.session_state.vector_store.similarity_search(user_question)
        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=st.secrets["GOOGLE_API_KEY"])
        template = "Use the context below to answer accurately.\nContext: {context}\nQuestion: {question}\nAnswer:"
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.chat_message("assistant").write(response["output_text"])
