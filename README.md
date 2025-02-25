# Chatbot with LangChain, Ollama, Streamlit, and Groq

This project is a chatbot built using LangChain, Ollama, Python, Streamlit, and Groq. It leverages Retrieval-Augmented Generation (RAG) and Graph Database capabilities to provide intelligent, context-aware responses.

 ### Features
- Retrieval-Augmented Generation (RAG): Combines retrieval-based and generative models for accurate and contextually relevant responses.

- Graph Database Integration: Utilizes a graph database for structured data storage and retrieval.

- Streamlit Interface: Provides a user-friendly web interface for interacting with the chatbot.

- Ollama and Groq: Powers the chatbot with advanced language models and efficient computation.

### Technologies Used
- LangChain: Framework for building applications powered by language models.

- Ollama: Platform for running and fine-tuning large language models.

- Python: Primary programming language for backend logic.

- Streamlit: Open-source framework for building interactive web applications.

- Groq: High-performance computing platform for AI workloads.

- Graph Database: Used for storing and querying structured data (e.g., Neo4j, Amazon Neptune, or similar).

### Getting Started
#### Prerequisites
Before running the chatbot, ensure you have the following installed:

- Python 3.8+
- Pip (Python package manager)
- Ollama (set up and running locally or remotely)
- Groq API Key (if using Groq for computation)
- Graph Database (e.g., Neo4j, Amazon Neptune, or similar)

### Installation
1. Clone the repository:

   ```bash
   git clone https://github.com/alekhya-p/gen_ai.git
   cd gen_ai
   ```

2. Install the required Python packages:

    ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
    - Create a .env file in the root directory.
    - Add your Groq API key and other sensitive information:
   ```bash
    GROQ_API_KEY=your_groq_api_key
    GRAPH_DB_URL=your_graph_db_url
    GRAPH_DB_USER=your_graph_db_user
    GRAPH_DB_PASSWORD=your_graph_db_password
   ```
    

4. Running the Chatbot
    - Start the Streamlit application:
     ```bash
      streamlit run app.py
      ```
    - Open your browser and navigate to http://localhost:8501 to interact with the chatbot.

### How It Works
Retrieval-Augmented Generation (RAG)
The chatbot uses RAG to combine:

- **Retrieval**: Fetches relevant information from a knowledge base or graph database.
- **Generation**: Uses a language model (e.g., Ollama) to generate responses based on the retrieved data.

#### Graph Database Integration
The graph database stores structured data, such as entities and relationships, which the chatbot queries to provide accurate and context-aware responses.

#### Streamlit Interface
The Streamlit app provides a simple and intuitive interface for users to interact with the chatbot. Users can input queries and view responses in real-time.

**Project Structure**
- **`app.py`**: The main Streamlit application that hosts the chatbot interface.
- **`rag.py`**: Implements Retrieval-Augmented Generation (RAG) for the chatbot.
- **`graph_db.py`**: Handles integration with the graph database for structured data storage and retrieval.
- **`requirements.txt`**: Lists all Python dependencies required to run the project.
- **`.env`**: Stores environment variables (e.g., API keys, database credentials). Add your own sensitive information here.
- **`.gitignore`**: Specifies files and directories to be ignored by Git (e.g., `.env`, virtual environments).
- **`README.md`**: Provides an overview of the project, setup instructions, and usage guidelines.

### Customization
- **Language Model**: Replace Ollama with another model (e.g., OpenAI GPT, Hugging Face models).

- **Graph Database**: Use a different graph database (e.g., Neo4j, Amazon Neptune).

- **Streamlit UI**: Customize the Streamlit interface to match your branding or add new features.

Chatbot using Ollama in local
![Screen](images/image.png)

RAG Implementation
![alt text](images/image-1.png)

Graph_DB
![alt text](images/graph_db.png)

