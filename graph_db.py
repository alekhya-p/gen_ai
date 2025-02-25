from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.schema import Document
from langchain_neo4j import Neo4jGraph
from langchain.prompts import PromptTemplate
import pdfplumber
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
import os
import re
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)

llm_engine = ChatGroq(api_key=GROQ_API_KEY,model_name="llama-3.3-70b-versatile")


PDF_DOCUMENT_PATH = "document_store/pdfs/"

# Ensure the directory exists
os.makedirs(PDF_DOCUMENT_PATH, exist_ok=True)

# Save uploaded file
def save_uploaded_file(uploaded_pdf):
    file_path = os.path.join(PDF_DOCUMENT_PATH, uploaded_pdf.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())
    return file_path


# Extract text from PDF
def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Split text into chunks
def chunk_documents(text):
    
    # Preprocess the text to remove sequences of dots
    cleaned_text = re.sub(r"\.{2,}", "", text)
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_text(text)
    
    # Print chunks for debugging (optional)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:\n{chunk}\n{'-'*50}")
    
    # Create Document objects with metadata
    documents = []
    for i, chunk in enumerate(chunks):
        # Create a Document object for each chunk
        doc = Document(
            page_content=chunk,  # Use the chunk text directly
            metadata={
                'chunk_index': i + 1,  # Add chunk index for traceability
            }
        )
        documents.append(doc)
    
    return documents

# Initialize LLM graph transformer
llm_transformer = LLMGraphTransformer(
    llm=llm_engine,
    node_properties=False, 
    relationship_properties=False)

# Convert documents to graph documents
def file_to_graph(documents):
    try:
        graph_documents = llm_transformer.convert_to_graph_documents(documents)
        return graph_documents
    except Exception as e:
        print(f"Error in convert_to_graph_documents: {e}")
        raise



#Main workflow

if __name__ == "__main__":
    
    st.title("PDF to Knowledge Graph")
    
    with st.sidebar:
        uploaded_doc = st.file_uploader(
        "Upload a document(PDF)",
        type="pdf",
        help="Select a pdf document",
        accept_multiple_files=False
        )
    
        if uploaded_doc:
            # Save the uploaded file
            save_file = save_uploaded_file(uploaded_doc)
            st.success(f"File saved to: {save_file}")

            # Extract text from PDF
            raw_text = extract_text_from_pdf(save_file)
            st.write("Text extracted from PDF.")

            # Split text into chunks
            processed_chunks = chunk_documents(raw_text)
            st.write("Text split into chunks.")

            # Convert chunks to graph documents
            graph_documents = file_to_graph(processed_chunks)
            st.write("Documents converted to graph documents.")
            
            # Extract nodes and relationships
            nodes = [{"id": node.id, "type": node.type, "properties": node.properties} for node in graph_documents[0].nodes]
            relationships = [{"source": rel.source.id, "target": rel.target.id, "type": rel.type, "properties": rel.properties} for rel in graph_documents[0].relationships]
            
            st.write("Nodes:", nodes)
            st.write("Relationships:", relationships)
            
            # Construct Cypher query
            cypher_query = """
            // Create nodes
            UNWIND $nodes AS node
            MERGE (n:Node {id: node.id})
            SET n += node.properties
            SET n:`${node.type}`

            // Use WITH to separate node creation from relationship creation
            WITH $relationships AS rels

            // Create relationships
            UNWIND rels AS rel
            MATCH (source {id: rel.source}), (target {id: rel.target})
            MERGE (source)-[r:`${rel.type}`]->(target)
            SET r += rel.properties
            """

            # Execute the query
            try:
                result = graph.query(cypher_query, params={"nodes": nodes, "relationships": relationships})
                st.write("Data pushed to Neo4j successfully!")
                st.success("Document processed successfully. Ask your question below.")
            except Exception as e:
                st.error(f"Error pushing data to Neo4j: {e}")
            
user_input = st.chat_input("Enter your question about your document")

if user_input:
    # Query Neo4j based on user input
    query = f"""
    MATCH (n)
    WHERE n.name CONTAINS '{user_input}' OR n.description CONTAINS '{user_input}'
    RETURN n
    """
    result = graph.query(query)
    st.write("Query Result:", result)
    
    


    
    


    





