import os  # Add this line to import the os module
import pinecone
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV,
)

# giving the index a name
index_name = pinecone.Index(INDEX_NAME)

delete_response = index_name.delete(deleteAll=True)
