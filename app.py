import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Rest of your code follows...



# 1. Setup Gemini
genai_key = st.secrets["GOOGLE_API_KEY"]

# 2. Extract Text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# 3. Split Text into Chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# 4. Create Vector Store (The AI's Memory)
def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

# 5. The Chat Logic
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the context, just say "The provided KRA documents do not contain this information."
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# --- UI LOGIC ---
st.set_page_config(page_title="TaxRafiki AI", page_icon="🇰🇪")
st.title("🇰🇪 TaxRafiki: Your KRA AI Guide")

with st.sidebar:
    st.title("Menu:")
    pdf_docs = st.file_uploader("Upload KRA PDFs", accept_multiple_files=True)
    if st.button("Submit & Process"):
        with st.spinner("Analyzing Laws..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            st.session_state.vector_store = get_vectorstore(text_chunks)
            st.success("Ready! Ask your questions.")

user_question = st.chat_input("Ask about Kenyan Taxes...")

if user_question:
    if "vector_store" not in st.session_state:
        st.error("Please upload and process PDFs first!")
    else:
        docs = st.session_state.vector_store.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        
        with st.chat_message("user"):
            st.write(user_question)
        with st.chat_message("assistant"):
            st.write(response["output_text"])
