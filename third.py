
import streamlit as st
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader, UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import tempfile

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to save uploaded files temporarily
def save_uploaded_file(uploaded_file):
    temp_file_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())
    return temp_file_path

# Function to extract text from different document types
def load_documents(doc_files):
    texts = []
    sources = []
    
    for doc in doc_files:
        ext = os.path.splitext(doc.name)[1].lower()
        temp_path = save_uploaded_file(doc)  # Save file temporarily
        
        if ext == ".pdf":
            loader = PyPDFLoader(temp_path)
        elif ext == ".pptx":
            loader = UnstructuredPowerPointLoader(temp_path)
        elif ext == ".docx":
            loader = UnstructuredWordDocumentLoader(temp_path)
        elif ext == ".txt":
            loader = TextLoader(temp_path)
        else:
            st.error(f"Unsupported file type: {ext}")
            continue
        
        loaded_docs = loader.load()
        for ld in loaded_docs:
            texts.append(ld.page_content)
            sources.append(doc.name)
    
    return texts, sources

# Function to handle text chunks for Vector Store with source
def get_text_chunks(texts, sources):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    chunk_sources = []
    
    for text, source in zip(texts, sources):
        split_chunks = text_splitter.split_text(text)
        chunks.extend(split_chunks)
        chunk_sources.extend([source] * len(split_chunks))
    
    return chunks, chunk_sources

# Function to get Vector Store
def get_vector_store(text_chunks, chunk_sources):
    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # FAISS automatically handles embedding generation via the `from_texts` method
    vector_store = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        metadatas=[{"source": src} for src in chunk_sources]
    )
    
    # Save the vector store locally
    vector_store.save_local("faiss_index")
    return vector_store

# Function to initialize conversational chain
def get_conversational_chain(vector_store):
    if vector_store is None:
        raise ValueError("Vector store is not initialized.")
    
    prompt_template = """
        You are a helpful assistant capable of handling both summarization and Q&A tasks based on document context and previous conversations. Here's how you should respond based on the user's request:

    If the task is to **summarize**:
    - Provide a concise summary of the document in no more than 200 words.
    - Use clear headings and bullet points to structure the summary for readability.
    
    If the task is to **answer a question**:
    - Use the context from the document to answer the user's question.
    - Connect the response to the ongoing conversation history, making the conversation interactive.
    
    
    In both cases:
    - Respond in clear, simple language that is easy for non-experts to understand.
    - Include the source of the document from which the information is derived.

    Below is the context and chat history that you should use:

    ------
    <ctx>
    {context}
    </ctx>
    ------
    <hs>
    {history}
    </hs>
    ------
    {question}

    Your response should:
    1. Be clear and accessible to a layman.
    2. Provide the source document.
    """


    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1)
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["history", "context", "question"]
    )

    memory = ConversationBufferMemory(
        memory_key="history",
        input_key="question"
    )

    # Check for vector store
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    retrieval_chain = RetrievalQA.from_chain_type(llm=model,
                                                  chain_type='stuff',
                                                  retriever=retriever,
                                                  chain_type_kwargs={
                                                      "prompt": prompt,
                                                      "memory": memory
                                                  })

    return retrieval_chain


# Function to handle user input and response
def user_input(user_question, vector_store):
    chain = get_conversational_chain(vector_store)
    
    response = chain(
        {"query": user_question}, return_only_outputs=True
    )

    # Debug: Print the response to see its structure
    #st.write("Debug - Full Response: ", response)

    if "result" in response:
        st.write("Reply (Source): ", response["result"])
    else:
        st.write("No valid response received.")

# Main function to run Streamlit app
# Main function to run Streamlit app
def main():
    st.set_page_config(page_title="Chat with Documents using Gemini")
    st.header("Chat or Summarize Your Documents 💁")

    # Maintain vector store and responses in session
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    
    if "response" not in st.session_state:
        st.session_state.response = ""

    # Two functionalities: Q&A or Summarization
    functionality = st.selectbox("Choose Functionality:", options=["Q&A", "Summarize"], index=0)

    # Ask a question if functionality is Q&A
    user_question = ""
    if functionality == "Q&A":
        user_question = st.text_input("Ask a Question from the Documents", value=st.session_state.response)
    
    if st.button("Ask Question"):
       st.session_state.response = user_question
       user_input(user_question, st.session_state.vector_store)  # Process the question here when button is clicked

    # Button to clear the response and user input
    if st.button("Clear Response"):
        st.session_state.response = ""
        st.write("Response cleared.")
        # Version 1.36 and below
        if st.__version__ <= "1.36.0":
           st.experimental_rerun()  # Works for older versions
        # Version 1.37 and above
        else:
            st.rerun()  # Use st.rerun() for newer versions

    # Summarization functionality
    if functionality == "Summarize":
        if st.button("Summarize Document"):
            user_input("Summarize the document", st.session_state.vector_store)

    # Document uploader and processing in sidebar
    with st.sidebar:
        st.title("Menu:")
        doc_files = st.file_uploader("Upload your Documents (PDF, PPTX, DOCX, TXT)", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_texts, sources = load_documents(doc_files)
                text_chunks, chunk_sources = get_text_chunks(raw_texts, sources)
                st.session_state.vector_store = get_vector_store(text_chunks, chunk_sources)
                st.success("Done")

if __name__ == "__main__":
    main()