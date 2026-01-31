import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

# 1. FIXED Google Drive Connection
def get_drive_service():
    # This pulls your secrets and cleans the mobile newline bug
    info = dict(st.secrets["gcp_service_account"])
    info["private_key"] = info["private_key"].replace("\\n", "\n")
    creds = service_account.Credentials.from_service_account_info(info)
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
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=st.secrets["GOOGLE_API_KEY"])
    return FAISS.from_texts(chunks, embedding=embeddings)

# --- UI LOGIC ---
st.set_page_config(page_title="TaxRafiki AI", page_icon="🇰🇪")
st.title("🇰🇪 TaxRafiki: Auto-Sync KRA Guide")

FOLDER_ID = "11gCstGrg63yaIH2DTfsEfq6zEl1bRzQp"

with st.sidebar:
    if st.button("🔄 Sync Latest Laws"):
        with st.spinner("Syncing..."):
            try:
                raw_text = download_pdfs_from_drive(FOLDER_ID)
                if raw_text:
                    st.session_state.vector_store = get_vectorstore(raw_text)
                    st.success("Synced!")
                else:
                    st.warning("No PDFs found.")
            except Exception as e:
                st.error(f"Error: {e}")

user_question = st.chat_input("Ask a tax question...")
if user_question:
    if "vector_store" not in st.session_state:
        st.error("Please Sync first!")
    else:
        docs = st.session_state.vector_store.similarity_search(user_question)
        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=st.secrets["AIzaSyDAGBiZ7zOSDqHT8xiufexKxOM9gp4YWx4"])
        prompt = PromptTemplate(template="Context:\n{context}\nQuestion:\n{question}\nAnswer:", input_variables=["context", "question"])
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.chat_message("assistant").write(response["output_text"])
