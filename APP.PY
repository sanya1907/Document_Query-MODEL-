import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

# Function to create vector embeddings
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")  # Data Ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)  # Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])  # Splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector Embeddings

# Custom CSS for styling
zoom_level = st.sidebar.slider("Zoom Level", 0.5, 2.0, 1.0, 0.1)

st.markdown(f"""
    <style>
        .main {{
            background-color: #f9fafb;
            font-family: 'Arial', sans-serif;
            zoom: {zoom_level};
        }}
        .stButton button {{
            background-color: #4a90e2;
            color: white;
            border-radius: 8px;
            padding: 12px 24px;
            font-size: 18px;
            border: none;
        }}
        .stButton button:hover {{
            background-color: #357ABD;
        }}
        .stTextInput > div > input {{
            padding: 12px;
            font-size: 18px;
            border-radius: 8px;
            border: 1px solid #4a90e2;
        }}
        .stTextInput > div > input:focus {{
            border-color: #357ABD;
        }}
        .stExpanderHeader {{
            font-weight: bold;
            font-size: 20px;
            color: #4a90e2;
        }}
        .highlight {{
            background-color: #e3f2fd;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #4a90e2;
            color: #4a90e2;
            margin-bottom: 10px;
        }}
        .response {{
            background-color: #ffffff;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #dddddd;
            color: #333333;
        }}
        .stSidebar {{
            background-color: #4a90e2;
            color: white;
        }}
        .stSidebar .sidebar-content a {{
            color: white;
        }}
        .stSidebar .sidebar-content a:hover {{
            color: #a2c9ff;
        }}
    </style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False

# Page content based on feedback submission
if st.session_state.feedback_submitted:
    # Thank You Page
    st.title("Thank You!")
    st.markdown("""
        <p style="font-size: 24px; color: #333333;">Thank you for using our document query tool!</p>
        <p style="font-size: 18px; color: #666666;">We acknowledge your feedback and appreciate your input. It helps us improve our services.</p>
    """, unsafe_allow_html=True)

    # "Back" Button
    if st.button("Back to Main Page"):
        st.session_state.feedback_submitted = False

else:
    # Main Page
    st.title("Document Query Using NVIDIA AI")
    st.subheader("Leverage NVIDIA's advanced AI models to query your documents effectively.")
    st.markdown("""
        <p style="font-size: 22px; color: #333333;">Welcome to the Document Query tool! This application uses NVIDIA's powerful AI models to answer questions based on the content of your uploaded documents.</p>
    """, unsafe_allow_html=True)

    # Sidebar for additional options
    st.sidebar.title("Options")
    st.sidebar.info("Adjust settings or view more information about the demo.")

    # Sidebar: Information about NVIDIA NIM API Key
    st.sidebar.subheader("About NVIDIA NIM API Key")
    st.sidebar.markdown("""
        The NVIDIA NIM API Key is an authentication token that allows access to NVIDIA's AI models and services. 
        It enables advanced capabilities such as natural language processing, text embeddings, and question answering. 
        By integrating with NVIDIA's powerful models, you can leverage high-quality AI features to enhance your applications.
    """)

    # Sidebar: Feedback Page
    feedback = st.sidebar.text_area("Leave your feedback or report issues here:")

    if st.sidebar.button("Submit Feedback"):
        if feedback:
            st.session_state.feedback = feedback
            st.session_state.feedback_submitted = True
        else:
            st.sidebar.warning("Please enter your feedback before submitting.")

    # File uploader (moved above Step 1)
    uploaded_file = st.file_uploader("Upload your PDF documents", type="pdf")

    # Main content with columns
    col1, col2 = st.columns([2, 1])

    with col1:
        # Document Embedding
        st.header("Step 1: Document Embedding")
        st.write("Click the button below to create a vector database from your documents.")

        # Embedding documents
        if st.button("Generate Embeddings"):
            if uploaded_file:
                # Use the uploaded file if present
                st.session_state.loader = PyPDFDirectoryLoader(uploaded_file)  # Data Ingestion
                with st.spinner("Generating embeddings..."):
                    vector_embedding()
                    st.success("Vector Store DB is ready! You can now ask questions.")
            else:
                st.warning("Please upload a PDF document.")

    with col2:
        st.image("https://www.nvidia.com/content/dam/en-zz/Solutions/about-nvidia/logo-and-brand/02-nvidia-logo-color-blk-500x200-4c25-p@2x.png", caption="NVIDIA Logo", use_column_width=True)

    # Prompt input
    st.header("Step 2: Ask a Question")
    prompt1 = st.text_input("Enter Your Question From Documents", placeholder="Type your question here...")

    # Check if the text input is not empty and process the question
    if prompt1 and "vectors" in st.session_state:
        llm = ChatNVIDIA(model="meta/llama3-70b-instruct")
        prompt = ChatPromptTemplate.from_template(
            """
            Answer the questions based on the provided context only.
            Please provide the most accurate response based on the question
            <context>
            {context}
            <context>
            Questions:{input}
            """
        )
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        response_time = time.process_time() - start

        # Displaying results
        st.subheader("Here is your answer:")
        st.markdown(f'<div class="response">{response["answer"]}</div>', unsafe_allow_html=True)
        st.write(f"Response time: {response_time:.2f} seconds")

        # Document similarity search results
        with st.expander("View Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.markdown(f'<div class="highlight">{doc.page_content}</div>', unsafe_allow_html=True)
                st.write("<hr>", unsafe_allow_html=True)

        # Display a progress bar
        with st.spinner("Processing..."):
            time.sleep(1)
