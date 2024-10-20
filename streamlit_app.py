import os
import time
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings   
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
import shutil  # Import shutil to remove the directory

# Function to load and embed text from a PDF
def load_and_embed_pdf(uploaded_file, db_path):
    try:
        start_time = time.time()
        # Text extraction from PDF
        reader = PdfReader(uploaded_file)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        if not text:
            st.error("No text extracted from the document.")
            return False

        # Text chunking for embeddings
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
        chunks = text_splitter.split_text(text)

        # Embed and store using Chroma
        embedding = OpenAIEmbeddings(model="text-embedding-3-small")
        vectordb = Chroma.from_texts(texts=chunks, embedding=embedding, persist_directory=db_path)
        embedding_time = time.time()
        print(f"Total processing time: {embedding_time - start_time:.2f} seconds")
        return True
    except Exception as e:
        st.error(f"An error occurred while processing the PDF: {e}")
        return False

# Function to retrieve answers based on a query
def get_answer(query, db_path):
    try:
        # Load the persisted vector store
        vectordb = Chroma(persist_directory=db_path, embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"))
        docs = vectordb.similarity_search(query, k=15)  # Adjust 'k' to limit the number of retrieved documents

        # Create a prompt template for generating the response
        prompt_template = PromptTemplate(
            input_variables=['query', 'context'],
            template="""You are an expert assistant for question-answering.
            Here, a context related to religious documents like holy books, stories, prayers, or research papers will be provided.
            Your task is to read, understand, and analyze the context before answering the question.
            Feel free to provide concise, human-friendly, and clear answers.

            Note: If you don't know the answer, just say that you don't know. Don't make assumptions.

            Given the context below, answer the question:
            Context: {context}
            Question: {query}
            Answer:"""
        )
        
        # Initialize the LLM with the appropriate model and parameters
        llm = ChatOpenAI(model='gpt-4o', temperature=0, max_tokens=200)
        chain = LLMChain(llm=llm, prompt=prompt_template)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Generate the answer
        return docs, chain.run(query=query, context=context)
    except Exception as e:
        st.error(f"An error occurred while retrieving the answer: {e}")
        docs = []
        return docs, "Unable to process the request."

def remove_chroma_db(db_path):
        try:
            if os.path.exists(db_path):
                shutil.rmtree(db_path)
                print("Chroma database removed.")
                return True
        except PermissionError:
            print("Attempt: Unable to remove the Chroma database. It may be in use.")
            return False


# Streamlit UI
st.markdown("<h1 style='text-align: center;'>GRANTH-RAG</h1>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center;'>Here you can ask questions about your religious documents</h6>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload your document here (.pdf)", type="pdf")

os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
# Define the file paths
save_path = "File"
specific_filename = "granth-rag.pdf"
file_path = os.path.join(save_path, specific_filename)
db_path = 'chroma_db'

# Initialize session state for processing flag
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False

# Save the uploaded file and process it
if uploaded_file:
    # Remove the existing Chroma database if it exists
    # if os.path.exists(db_path):
    #     remove_chroma_db(db_path)

    # os.makedirs(save_path, exist_ok=True)
    # with open(file_path, "wb") as f:
    #     f.write(uploaded_file.getbuffer())
    
    st.info("Processing the uploaded PDF...")
    success = load_and_embed_pdf(uploaded_file, db_path)
    
    if success:
        st.session_state.pdf_processed = True  # Set the flag to true
        st.success("PDF processed and embedded successfully. You can now ask questions.")
    else:
        st.error("Failed to process the PDF. Please check the document format or page range.")

# Get user input for a question
question = st.text_input("Ask a question", placeholder="Type your question here...")

# Handle the submission of the question
if st.button("Submit"):
    if st.session_state.pdf_processed:  # Only allow question submission if PDF is processed
        if question:
            docs, answer = get_answer(question, db_path)
            print(docs)
            st.write(answer)
        else:
            st.warning("Please enter a question to proceed.")
    else:
        st.warning("Please upload and process a PDF file first.")