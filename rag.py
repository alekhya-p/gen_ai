import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader #is used to load and process PDF files in LangChain.
from langchain_text_splitters import RecursiveCharacterTextSplitter # used to split large text documents into smaller chunks
from langchain_core.vectorstores import InMemoryVectorStore #used for storing small datasets in RAM
# from langchain_ollama import OllamaEmbeddings #used to convert text into vector
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS #widely used for in-memory embeddings.



PROMPT_TEMPLATE = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

Query: {user_query}
Context: {document_context}

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Do not include any external knowledge or assumptions not present in the given text.
"""

PDF_DOCUMENT_PATH = "document_store/pdfs/"
EMBEDDING_MODEL=OllamaEmbeddings(model="nomic-embed-text")
DOCUMENT_VECTOR_DB=InMemoryVectorStore(embedding=EMBEDDING_MODEL)
LANGUAGE_MODEL=OllamaLLM(model="llama3.2:3b")

def save_uploaded_file(uploaded_file):
    file_path = PDF_DOCUMENT_PATH + uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    text_processor=RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""]
    )
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)
    
def find_related_document(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

def generate_answer(user_query, context_document):
    context_text = [doc.page_content for doc in context_document]
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})

#UI Configuration
st.title("Document Mind AI")
st.markdown("Your are Intelligent Document Assistant")

with st.sidebar:
    uploaded_doc = st.file_uploader(
    "Upload a document(PDF)",
    type="pdf",
    help="Select a pdf document",
    accept_multiple_files=False
    )
    
    if uploaded_doc:
        save_file = save_uploaded_file(uploaded_doc)
        raw_file = load_documents(save_file)
        processed_chunks = chunk_documents(raw_file)
        st.write(processed_chunks)
        index_documents(processed_chunks)
        
        st.success("Document processed successfully. Ask your question below")


user_input = st.chat_input("Enter your question about the document")
    
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
        
    with st.spinner("Prosessing.."):
        relevent_file = find_related_document(user_input)
        ai_response = generate_answer(user_input, relevent_file)
        
    with st.chat_message('assistant'):
        st.write(ai_response)