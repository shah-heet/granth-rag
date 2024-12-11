import pandas as pd
from llama_index.core.evaluation import DatasetGenerator, RelevancyEvaluator
from llama_index.core import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

import nest_asyncio
nest_asyncio.apply()

import os
from dotenv import load_dotenv
load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

def load_documents_from_directory(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):  # adjust file extension as needed
            with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as file:
                documents.append(file.read())
    return documents

# Function to perform semantic chunking 
def semantic_chunking(text, model_name,breakpoint_threshold_type, breakpoint_threshold_amount):
    embed_model = HuggingFaceEmbeddings(model_name=model_name)
    semantic_chunker = SemanticChunker(embed_model, 
                                       breakpoint_threshold_type=breakpoint_threshold_type,
                                       breakpoint_threshold_amount = breakpoint_threshold_amount)#,sentence_split_regex=r"[.?!]\s*"
    
    docs = semantic_chunker.create_documents(text)

    # Convert Langchain documents to LlamaIndex documents
    llama_docs = [Document(text=doc.page_content) for doc in docs]
    return llama_docs

#vectordb creation and store the chunks
def create_vector_db(model_name,chunks,persist_directory,):
    # Define the directory where ChromaDB will store data
    persist_directory = persist_directory

    # Initialize the embedding function with trust_remote_code=True
    embedding_function = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"trust_remote_code": True}  # This allows loading custom model code
    )

    # Create ChromaDB instance with the embedding function
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function
    )

    # Alternative Option 2: If you want to preserve metadata
    vectordb.add_texts(
        texts=[chunk.get_content() for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks]
    )
    
    # Persist the database locally
    vectordb.persist()

    print(f"ChromaDB has been successfully saved at: {persist_directory}")

#loading created vector db
def load_vector_db(model_name, persist_directory):

    # Initialize the embedding function with trust_remote_code=True
    embedding_function = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"trust_remote_code": True}  # This allows loading custom model code
    )

    # Load the existing ChromaDB instance
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function
    )

    print(f"ChromaDB has been successfully loaded from: {persist_directory}")
    return vectordb

def get_similar_chunks(question,vectordb):
    similar_chunks = vectordb.similarity_search(question,k=10)
    return similar_chunks

def get_answer(model_name,query,similar_chunks):

    prompt_template = PromptTemplate(
        input_variables=['query', 'context'],
        template="""
        You are an expert assistant specializing in question-answering. The provided context will be from religious texts such as holy books, stories, prayers, or research papers.

        Your task is to carefully read, understand, and analyze the given context before responding to the user's questions. Ensure your answers are accurate, concise, and human-friendly, while remaining clear and to the point.

        Important:
        Do not make assumptions or fabricate responses.

        Based on the context provided below, answer the question:

        Context: {context}

        Question: {question}

        Answer:
        """
    )


    llm = ChatOpenAI(model=model_name,temperature=0,max_tokens=1500)#
    chain = LLMChain(llm=llm, prompt=prompt_template)

    return chain.run(question=query, context=similar_chunks)

def excel_fill(file_name, eval_questions, vectordb, model_name):
    # File name and column definitions
    file_name = file_name
    columns = ['index', 'question', 'answer', 'source']

    # Check if the file exists
    if os.path.exists(file_name):
        # Load existing file
        df = pd.read_excel(file_name)
    else:
        # Create a new DataFrame with specified columns
        df = pd.DataFrame(columns=columns)

    # Update the DataFrame for each index
    for i in range(len(eval_questions)):
        sim_chunks = get_similar_chunks(eval_questions[i], vectordb)
        ansr = get_answer(model_name, eval_questions[i], sim_chunks)

        new_row = {
            'index': i,
            'question': eval_questions[i],
            'answer': ansr,
            'source': sim_chunks
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Save the updated DataFrame to the Excel file
    df.to_excel(file_name, index=False)
    print(f"Data successfully updated in {file_name}.")

def main():
    # Load documents
    directory_path = r"D:\Anand\ra-granth\temp_text_file"
    documents = load_documents_from_directory(directory_path)
    documents = [documents[0][:5000]]  

    print("Semantic chunking started...")

    # Chunk the documents semantically
    text = documents
    chunking_model_name = "mixedbread-ai/mxbai-embed-large-v1"
    breakpoint_threshold_type = "percentile"
    breakpoint_threshold_amount = 0.8

    semantic_chunks = semantic_chunking(text, chunking_model_name, breakpoint_threshold_type, breakpoint_threshold_amount)
    print("__________________", semantic_chunks[:3], "__________________")
    print("Semantic chunking done...")

    # Question generation (llama_index)
    data_generator = DatasetGenerator.from_documents(semantic_chunks)
    eval_questions = data_generator.generate_questions_from_nodes(10)
    print(eval_questions)

    # VectorDB creation and store the chunks
    persist_directory = f"D:\\Anand\\ra-granth\\test_db\\chroma_db_{breakpoint_threshold_amount}"
    chunks = semantic_chunks
    model_name = "Alibaba-NLP/gte-base-en-v1.5" 

    create_vector_db(model_name, chunks, persist_directory)

    print("Creating dataset generator...")

    # Loading created vector db
    vectordb = load_vector_db(model_name, persist_directory)

    #question = eval_questions[2]
    #similar_chunks = get_similar_chunks(question, vectordb)

    #ansr = get_answer("gpt-4o-mini", question, similar_chunks)
    #print("Answer")
    #print("''''''", ansr, "''''''")

    file_name = 'qna.xlsx'
    model_name = "gpt-4o-mini"
    excel_fill(file_name, eval_questions, vectordb, model_name)

if __name__ == "__main__":
    main()

