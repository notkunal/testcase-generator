import streamlit as st
import PyPDF2
import requests
import os
import tempfile
import re
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Initialize the Google Gemini 1.5 Flash model
def init_llm(api_key):
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        api_key=api_key,
        temperature=0.8,
        max_tokens=500,
        timeout=15,
        max_retries=3
    )

# Extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(reader.pages)):
        text += reader.pages[page_num].extract_text()
    return text

# Download PDF from arXiv link
def download_pdf(arxiv_url):
    pdf_url = arxiv_url.replace("abs", "pdf") + ".pdf"
    response = requests.get(pdf_url)
    if response.status_code == 200:
        return response.content
    else:
        st.error(f"Failed to download PDF from {pdf_url}")
        return None

# Save PDF to a temporary file
def save_pdf(content):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(content)
        return tmp_file.name

# Setup FAISS VectorStore for document retrieval
def create_faiss_index(texts):
    embeddings = HuggingFaceEmbeddings()
    faiss_index = FAISS.from_texts(texts, embeddings)
    return faiss_index

# Streamlit app
st.title("AI-Powered ArXiv and Document Research Assistant")

st.write(
    """
    This tool allows you to input multiple arXiv links or upload multiple PDFs, and then ask questions based on the contents of those documents.
    It uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from the documents and enables users to query the content and receive accurate, context-aware responses.

    Just like a chatbot, this tool keeps your question-response history intact and visible to you as well!
    """
)

# Store data in session state to persist across interactions
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'all_texts' not in st.session_state:
    st.session_state.all_texts = []
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'responses' not in st.session_state:
    st.session_state.responses = []  # Store responses across questions
if 'memory' not in st.session_state:
    # Initialize memory to track conversation context
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Function to reset session state
def reset_state():
    st.session_state.faiss_index = None
    st.session_state.all_texts = []
    st.session_state.llm = None
    st.session_state.responses = []  # Clear responses when resetting
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)  # Reset memory

# Choose input method: either arXiv links or PDF upload
input_method = st.radio("Choose input method:", ("ArXiv Links", "Upload PDFs"), on_change=reset_state)

if input_method == "ArXiv Links":
    with st.form(key='arxiv_form'):
        # Input arXiv links
        arxiv_links = st.text_area("Enter the arXiv links (one per line):").splitlines()
        submit_button = st.form_submit_button(label="Process Papers")

    if arxiv_links and submit_button:
        all_texts = []
        
        for link in arxiv_links:
            with st.spinner(f"Processing {link}..."):
                pdf_content = download_pdf(link)
                if pdf_content:
                    pdf_file_path = save_pdf(pdf_content)
                    with open(pdf_file_path, "rb") as pdf_file:
                        text = extract_text_from_pdf(pdf_file)
                        all_texts.append(text)
        
        if all_texts:
            # Create FAISS index with the text extracted from the papers
            st.write("Creating document retrieval index...")
            st.session_state.faiss_index = create_faiss_index(all_texts)
            st.session_state.all_texts = all_texts
            st.success("Index created successfully!")

elif input_method == "Upload PDFs":
    # Upload PDF documents
    uploaded_files = st.file_uploader("Upload PDF documents", accept_multiple_files=True, type=["pdf"])

    if uploaded_files and st.button("Process Uploaded PDFs"):
        all_texts = []
        
        for uploaded_file in uploaded_files:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                text = extract_text_from_pdf(uploaded_file)
                all_texts.append(text)
        
        if all_texts:
            # Create FAISS index with the text extracted from the uploaded PDFs
            st.write("Creating document retrieval index...")
            st.session_state.faiss_index = create_faiss_index(all_texts)
            st.session_state.all_texts = all_texts
            st.success("Index created successfully!")

# Function to separate LaTeX and text
def render_response(response):
    # Find LaTeX enclosed in $...$ or $$...$$ and render it separately
    parts = re.split(r'(\$.*?\$|\$\$.*?\$\$)', response)
    
    for part in parts:
        if part.startswith("$") and part.endswith("$"):  # LaTeX math
            st.latex(part.strip("$"))
        else:  # Regular text
            st.write(part)

# Function to handle question submission and answer generation
def handle_question(user_question):
    if user_question:  # Prevent empty submissions
        # Initialize the LLM and create a retrieval chain
        if st.session_state.llm is None:
            st.session_state.llm = init_llm("AIzaSyBoX4UUHV5FO4lvYwdkSz6R5nlxLadTHnU")  # Use your API key
        
        retriever = st.session_state.faiss_index.as_retriever()

        # Reformulate follow-up questions based on previous response, not entire chat history
        if len(st.session_state.responses) > 0:
            last_response = st.session_state.responses[-1]['answer']
            reformulated_question = f"The user is referring to your previous response: '{last_response}'. Now, they are asking: '{user_question}'."
        else:
            reformulated_question = user_question

        # Create a memory-aware retriever chain
        qa_chain = load_qa_chain(st.session_state.llm, chain_type="stuff")

        # Include memory in the chain
        chain = RetrievalQA(
            combine_documents_chain=qa_chain,
            retriever=retriever,
            memory=st.session_state.memory
        )

        # Answer the user's question using RAG with memory
        with st.spinner("Generating answer..."):
            try:
                # Save the user's question in memory
                st.session_state.memory.chat_memory.add_user_message(user_question)

                # Get response based on reformulated question
                response = chain.run(reformulated_question)
                
                # Save the assistant's response in memory
                st.session_state.memory.chat_memory.add_ai_message(response)

                # Append the new response to the list of responses
                st.session_state.responses.append({"question": user_question, "answer": response})

                return response  # Return response immediately to display it
            except Exception as e:
                st.error(f"An error occurred: {e}")
                return None

# Only show the question input and retrieval system if the FAISS index exists
if st.session_state.faiss_index:
    # Display all previous responses before the question input
    for idx, res in enumerate(st.session_state.responses):
        st.write(f"### Question {idx+1}: {res['question']}")
        st.write("**Answer:**")
        render_response(res['answer'])  # Handle LaTeX and regular text rendering

    # Inject JavaScript to detect Shift+Enter for new line and Enter for submit
    st.markdown("""
        <script>
        const textarea = document.querySelector("textarea");
        textarea.addEventListener("keydown", function(event) {
            if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                document.querySelector("button[aria-label='Generate Answer']").click();
            }
        });
        </script>
        """, unsafe_allow_html=True)

    # User question input, placed after displaying the responses
    user_question = st.text_area(
        "I can help you do further research based on the uploaded documents. Ask your queries based on the uploaded documents:",
        placeholder="Ask a question... (Press Enter to submit, Shift+Enter for new line)"
    )

    # Button to trigger the response generation
    if st.button("Generate Answer"):
        response = handle_question(user_question)
        if response:
            st.write(f"### Your Question: {user_question}")
            st.write("**Answer:**")
            render_response(response)