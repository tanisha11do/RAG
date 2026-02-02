# rag.py
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_classic.embeddings import HuggingFaceEmbeddings
from langchain_classic.llms import HuggingFacePipeline
from transformers import pipeline
#from langchain_openai import OpenAI

# Load OpenAI API key from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --------- Functions ---------

def load_pdfs(files):
    """
    Load PDF files and return a list of text chunks.
    """
    documents = []
    for file in files:
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        documents.append(text)

    # Split into smaller chunks for RAG
    text_splitters = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs_split = text_splitters.split_text(" ".join(documents))
    return docs_split


def create_vector_store(text_chunks):
    """
    Convert text chunks to embeddings and store in FAISS vectorstore.
    """
    #embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    return vector_store

'''
def get_answer(vector_store, query):
    """
    Retrieve relevant documents and answer a question using OpenAI LLM.
    """
    # Retrieve top 3 relevant chunks
    docs = vector_store.similarity_search(query, k=3)

    # Load QA chain
    llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
    chain = RetrievalQA(llm, chain_type="stuff")

    answer = chain.run(input_documents=docs, question=query)
    return answer '''


def get_answer(vector_store, query):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    hf_pipeline = pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Answer the question in 10-20 clear sentences only. "
            "Do not repeat the context. "
            "If the answer is not in the context, say 'I don't know.'"
        ),
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    return qa_chain.run(query)
