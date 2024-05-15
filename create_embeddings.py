import chromadb.utils.embedding_functions as embedding_functions
import chromadb

client = chromadb.HttpClient(host="localhost", port=29170, headers={"X-Chroma-Token":"jay"})
print(client.heartbeat())

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key="sk-proj-9f5L52hNZd3vlGBWS3deT3BlbkFJ9VLRlf6KdsbiFObt6syq",
                model_name="text-embedding-3-small"
            )

collection = client.get_or_create_collection("mf_collection", embedding_function=openai_ef)

def create_embeddings(chunks):
    """
    A function to create embeddings for the provided chunks.
    
    Args:
        chunks: A list of text chunks for which embeddings need to be created.
        
    Returns:
        None
    """
    ids = [f"id{i}" for i in range(len(chunks))]
    documents = chunks
    collection.upsert(documents=documents, ids=ids)
    print("Embeddings created successfully.")