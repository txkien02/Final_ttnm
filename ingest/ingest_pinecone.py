import os
import openai
import pinecone
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain

# Load environment variables from .env file
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

DATA_DIR = "./../data"


# Create vector database
def create_vector_database():
    """
    Creates a vector database using document loaders and embeddings.

    This function loads data from PDF, markdown and text files in the 'data/' directory,
    splits the loaded documents into chunks, transforms them into embeddings using HuggingFace,
    and finally persists the embeddings into a Chroma vector database.

    """
    
    def load_docs(directory):
        
        loader = DirectoryLoader(directory)
        documents = loader.load()
        return documents

    loaded_documents = load_docs(DATA_DIR)
    
    len(loaded_documents)
    # Split loaded documents into chunks
    def split_docs(documents, chunk_size=1000, chunk_overlap=20):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = text_splitter.split_documents(documents)
        return docs
    docs = split_docs(loaded_documents)
    # Initialize OpenAI embeddings
    embeddings =OpenAIEmbeddings(model_name="ada")



    # initialize pinecone
    pinecone.init(
        api_key=PINECONE_API_KEY,  # type: ignore
        environment=PINECONE_ENV,  # type: ignore
    )

    # giving index a name
    index_name = INDEX_NAME

    # delete index if same name already exists
    # if index_name in pinecone.list_indexes():
    #     pinecone.delete_index(index_name)

    # create index
    # pinecone.create_index(name=index_name, dimension=384, metric="cosine")

    # Create and store a pinecone vector database from the chunked documents
    index = Pinecone.from_documents(docs, embeddings, index_name=index_name)

if __name__ == "__main__":
    create_vector_database()
