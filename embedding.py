import chromadb.utils.embedding_functions as embedding_functions
import chromadb
from dotenv import load_dotenv
import os

load_dotenv()

host = os.getenv("CHROMADB_HOST")
port = os.getenv("CHROMADB_PORT")
token = os.getenv("CHROMADB_TOKEN")

client = chromadb.HttpClient(host=host, port=port, headers={"X-Chroma-Token":token})
print(client.heartbeat())

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key="sk-proj-9f5L52hNZd3vlGBWS3deT3BlbkFJ9VLRlf6KdsbiFObt6syq",
                model_name="text-embedding-3-small"
            )

collection = client.get_or_create_collection("mf_collection", embedding_function=openai_ef)

def get_embeddings():
    """
    A function to get embeddings from a collection.
    """
    return collection.get(include=['embeddings'])

def get_relevant_chunks():
    """
    A function to retrieve relevant text chunks based on a query text and the number of results to return.
    """
    results = collection.query(
    query_texts=["What is a Mutual Fund?"], # Chroma will embed this for you
    n_results=2 # how many results to return
    )
    return results
