import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings   
from langchain.vectorstores import Chroma # needed when we run this function in a separate cell
from langchain.embeddings import OpenAIEmbeddings   
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

def loading_pdf_and_embedding(file_path,db_path):
    #Text extraction form pdf    
    reader = PdfReader(file_path)
    text = ""
    #for page in reader.pages:
    #    text += page.extract_text()    

    start_page = 13  
    end_page = 18   

    for page_num in range(start_page, end_page + 1):
        page = reader.pages[page_num]
        text += page.extract_text() 

    #print(text)    

    #chunking
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap = 50)
    chunks=text_splitter.split_text(text)


    #embed and store
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    vectordb = Chroma.from_texts(texts=chunks,embedding=embedding,persist_directory=db_path)  

def get_answer(query):
    # Load the persisted vector store
    vectordbt = Chroma(persist_directory=db_path, embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"))
    

    docs = vectordbt.similarity_search(query)

    
    prompt_template = PromptTemplate(
        input_variables = ['query','context'],
        template="""You are an expert assistant for question-answering.
        Here in context religious document will be provided to you it can be any religious holybook,story , prayer ,research paper.
        your job is to read , understand and analyze the context before answering the question.
        user will ask u questions about the context and you have to answer them.
        feel free to answer the question briefly if needed, just Keep the answer human friendly,clear and to the point.

        Note : If you don't know the answer, just say that you don't know. Don't make assumptions.

        Given the context below,answer the question:\n\n
        Context: {context}\n\n
        Question: {question}\n\n
        Answer:
    """
    )
    

    llm = ChatOpenAI(model='gpt-4o-mini',temperature=0,max_tokens=200)#
    chain = LLMChain(llm=llm, prompt=prompt_template)

    return chain.run(question=query, context=docs)

# # # # # # #

#Streamlit code
# Title
st.markdown("<h1 style='text-align: center;'>GRANTH-RAG</h1>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center;'>Here you can question-answer on your religious 📄 document</h6>", unsafe_allow_html=True)
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]

    uploaded_file = st.file_uploader("Upload your document here (.pdf)", type="pdf")

    save_path = "File"
    specific_filename = "granth-rag.pdf"

    if uploaded_file:
        # Save the uploaded file to the specified path
        file_path = os.path.join(save_path, specific_filename)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        
        file_path = 'File/granth-rag.pdf'
        db_path = 'chroma_db'
        
        loading_pdf_and_embedding(file_path,db_path)

    question = st.text_input("Ask a question")

    if st.button("Submit"):
        if question:
            answer = get_answer(question)
            st.write(answer)
        else:
            st.write("Please upload a PDF and enter a question.")
