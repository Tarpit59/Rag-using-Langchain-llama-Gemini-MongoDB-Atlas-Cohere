from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_cohere import CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
import cohere
import markdown2
import re
import logging
import os

load_dotenv()

MONGODB_ATLAS_CLUSTER_URI = os.getenv('MONGODB_ATLAS_CLUSTER_URI')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
LLAMA_MODEL = "Llama3-8b-8192"
COHERE_RERANK_MODEL = 'rerank-english-v3.0'

mongodb_client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
DB_NAME = "langchain"
COLLECTION_NAME = "vectorSearch"
ATLAS_VECTOR_SEARCH_INDEX_NAME = 'vector_index'
MONGODB_COLLECTION = mongodb_client[DB_NAME][COLLECTION_NAME]

cohere_client = cohere.Client(COHERE_API_KEY)

embedding=GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model="models/embedding-001")
llm_gemini = ChatGoogleGenerativeAI(model='gemini-1.5-flash-001-tuning', temperature=0.2, google_api_key=GOOGLE_API_KEY, convert_system_message_to_human=True)
llm_llama = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=LLAMA_MODEL)

vector_store = MongoDBAtlasVectorSearch(
    collection=MONGODB_COLLECTION,
    embedding=embedding,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    relevance_score_fn="cosine",
)

def process_uploaded_pdfs(directory, chunk_size=5000, chunk_overlap=300):
    '''
    Input is all pdf files in directory
    Vectors are stored on Vector database
    '''
    file_loader = PyPDFDirectoryLoader(directory)
    documents = file_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    MONGODB_COLLECTION.delete_many({})
    vector_store.add_documents(chunks)

logging.basicConfig(level=logging.INFO)
def is_code_block(text):
    '''
    Input is text 
    Function identify the code block in the text
    '''
    return text.strip().startswith("```")

def clean_text_segment(text):
    '''
    Input is text
    Clean text segments while preserving mathematical symbols
    '''
    html_response = markdown2.markdown(text)
    clean_text = re.sub(r'<[^>]*>', '', html_response)
    return clean_text

def preserve_code_segment(code):
    '''
    Input is code
    Preserve code structure and comments'''
    code = re.sub(r'(\*\*|__)', '', code)  # Remove bold
    code = re.sub(r'(\*|_)', '', code)     # Remove italics
    return code.strip()

def process_response(response):
    '''
    Input is whole response
    Process mixed text and code content
    output is string after removing all formattings
    '''
    if not isinstance(response, str):
        logging.error(f"Invalid response type: {type(response)}. Expected a string.")
        return "Invalid response type."

    segments = re.split(r'(```[\s\S]*?```)', response)
    processed_segments = []

    for segment in segments:
        if is_code_block(segment):
            code_content = re.sub(r'```', '', segment).strip()
            processed_segments.append(preserve_code_segment(code_content))
        else:
            processed_segments.append(clean_text_segment(segment))
    
    return '\n'.join(processed_segments).replace('\n\n', '\n')

def get_rag_response(model_choice, question):
    '''
    Input is user model choice and the query
    output is query's answer
    '''
    if model_choice == 'google':
        llm = llm_gemini
    elif model_choice == 'llama':
        llm = llm_llama
    else:
        raise ValueError("Invalid model choice")
    
    retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 20, "score_threshold": 0.01},)
    
    compressor = CohereRerank(cohere_api_key= COHERE_API_KEY,model=COHERE_RERANK_MODEL)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, 
                                                           base_retriever=retriever)
    chain = RetrievalQA.from_chain_type(llm=llm, 
                                        retriever=compression_retriever)
    answer =chain({"query": question})
    answer = process_response(answer.get('result'))
    return answer