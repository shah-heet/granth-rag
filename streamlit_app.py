import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

def loading_pdf_and_embedding(file_path, db_path):
    # Text extraction from PDF    
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()   

    # Chunking the text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    chunks = text_splitter.split_text(text)

    # Embedding
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    # Creating FAISS index and saving
    vectordb = FAISS.from_texts(texts=chunks, embedding=embedding)
    vectordb.save_local(db_path)

    return True

def get_answer(query, db_path):
    # Load the FAISS index
    vectordbt = FAISS.load_local(db_path, OpenAIEmbeddings(model="text-embedding-3-large"), allow_dangerous_deserialization=True)
    docs = vectordbt.similarity_search(query)

    # Define the prompt template
    prompt_template = PromptTemplate(
        input_variables=['question', 'context'],
        template="""
        You are an expert assistant specializing in question-answering. The provided context will be from religious texts such as holy books, stories, prayers, or research papers.

        Your task is to carefully read, understand, and analyze the given context before responding to the user's questions. Ensure your answers are accurate, concise, and human-friendly, while remaining clear and to the point.

        Important:
        - If you don't know the answer, simply state, "I don't know." Do not make assumptions or fabricate responses.

        Based on the context provided below, answer the question:

        Context: {context}

        Question: {question}

        Answer:
        """
    )

    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0, max_tokens=1000)
    chain = LLMChain(llm=llm, prompt=prompt_template)

    # Combine the docs into a single context string
    context = "\n\n".join([doc.page_content for doc in docs])

    return chain.run(question=question, context=context)

# Streamlit code
# Title
st.markdown("<h1 style='text-align: center;'>GRANTH-RAG</h1>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center;'>Here you can question-answer on your religious document</h6>", unsafe_allow_html=True)

os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]

uploaded_file = st.file_uploader("Upload your document here (.pdf)", type="pdf")

save_path = "File"
specific_filename = "granth-rag.pdf"
file_path = 'File/granth-rag.pdf'
db_path = 'faiss_db'

if uploaded_file and not st.session_state.get('file_processed', False):
    # Save the uploaded file to the specified path
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, specific_filename)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Inform the user that the file is being processed
    with st.spinner('Processing the uploaded PDF...'):
        success = loading_pdf_and_embedding(file_path, db_path)
    
    if success:
        st.success("PDF processed and embedded successfully. You can now ask questions.")
        st.session_state.file_processed = True
    else:
        st.error("Failed to process the PDF. Please check the document format or content.")

question = st.text_input("Ask a question")

if st.button("Submit"):
    if question:
        answer = get_answer(question, db_path)
        st.write(answer)
    else:
        st.write("Please upload a PDF and enter a question.")