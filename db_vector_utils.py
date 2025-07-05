import numpy as np
from datetime import datetime
from tqdm import tqdm  # for progress tracking
import faiss


def batch_upload_to_pinecone(docs_after_split, hug_embedding_model, index, batch_size=100):
    """
    Upload documents to Pinecone in batches
    
    Args:
        docs_after_split: List of document chunks
        embedding_model: Model to create embeddings
        index: Pinecone index
        batch_size: Number of documents to process at once
    """
    total_docs = len(docs_after_split)
    for i in tqdm(range(0, total_docs, batch_size)):
        # Get the current batch
        batch_docs = docs_after_split[i:i + batch_size]
        
        # Prepare the batch data
        ids = [f"chunk-{i+j}" for j in range(len(batch_docs))]
        texts = [doc["text"] for doc in batch_docs]
        
        # Create embeddings for the entire batch at once
        #embeddings = embedding_model.encode(texts)

        embeddings = hug_embedding_model.embed_documents(texts)

        
        # Prepare vectors for the batch
        vectors = []
        for j, (doc, embedding) in enumerate(zip(batch_docs, embeddings)):
            vectors.append({
                "id": f"chunk-{i+j}",
                ##"values": embedding.tolist(),
                "values": embedding,
                "metadata": {
                    "text": doc["text"],
                    "source": doc["id"],
                            "timestamp": datetime.utcnow().isoformat()

                }
            })
            
        
        
        # Upload the batch to Pinecone
        index.upsert(vectors=vectors)
# Function to retrieve documents from Pinecone
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load a pre-trained QA model
qa_pipeline = pipeline("question-answering")

def retrieve_documents(query,index,embedding_model, top_k=5):
    # Encode the query into an embedding
    query_embedding = embedding_model.encode(query).tolist()

    # Use the updated argument order for `index.query()`
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

    # Extract and return the metadata (document text)
    return [res['metadata']['text'] for res in results['matches']]

# Function to answer a question using retrieved documents
def answer_question(query, top_k=5):
    # Step 1: Use your retrieve_documents function to perform Semantic Search
    documents = retrieve_documents(query, top_k=top_k)

    # Step 2: Combine the retrieved chunks into a single context
    context = " ".join(documents)

    # Step 3: Use the QA model to extract the answer
    result = qa_pipeline(question=query, context=context)

    # Handle low-confidence answers
    if result['score'] < 0.5:
        return "Sorry, I couldn't find a relevant answer in the documents."
    else:
        return result['answer']
    
def search_similar_documents(query, index, embedding_model, top_k=8):
    """
    Search for similar documents in Pinecone
    
    Args:
        query (str): Query string to match against documents
        index: Pinecone index object
        model: SentenceTransformer model
        top_k (int): Number of results to return
        
    Returns:
        list: List of similar documents with their scores
    """
    # Create query embedding
   # embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    #query_embedding = model.encode(query)
    query_embedding = embedding_model.embed_query(query)



    # Search in Pinecone
    results = index.query(
        #vector=query_embedding.tolist(),
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    # Format results
    formatted_results = []
    for match in results['matches']:
        formatted_results.append({
            'document': match['metadata']['text'],
            'score': match['score']
        })
    
    return formatted_results


# Example query
#query = "Which country has its capital in Washington?"
from datetime import datetime
from tqdm import tqdm
import numpy as np

def batch_upload_db_documents(
    docs_after_split,
    embedding_model,
    index,
    use_pinecone=True,
    batch_size=100,
    faiss_metadata_store=None
):
    """
    Uploads documents in batches to Pinecone or FAISS with optional metadata store.
    
    Args:
        docs_after_split: List of document chunks
        embedding_model: SentenceTransformer model
        index: FAISS or Pinecone index
        use_pinecone: True = Pinecone; False = FAISS
        batch_size: Number of documents per batch
        faiss_metadata_store: Optional list to store metadata (for FAISS only)
    """
    total_docs = len(docs_after_split)

    for i in tqdm(range(0, total_docs, batch_size)):
        batch_docs = docs_after_split[i:i + batch_size]
        texts = [doc["text"] for doc in batch_docs]
        embeddings = embedding_model.encode(texts)

        if use_pinecone:
            vectors = []
            for j, (doc, embedding) in enumerate(zip(batch_docs, embeddings)):
                vectors.append({
                    "id": f"chunk-{i+j}",
                    "values": embedding.tolist(),
                    "metadata": {
                        "text": doc["text"],
                        "source": doc["id"],
                        "timestamp": datetime.utcnow().isoformat()
                    }
                })
            index.upsert(vectors=vectors)

        else:
            # Add vectors to FAISS
            index.add(np.array(embeddings, dtype='float32'))

            # Store metadata separately in the provided list
            if faiss_metadata_store is not None:
                for doc in batch_docs:
                    faiss_metadata_store.append({
                        "text": doc["text"],
                        "source": doc["id"],
                        "timestamp": datetime.utcnow().isoformat()
                    })

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

def upload_docs_to_pinecone_via_vectorstore(docs_after_split, embedding_model, index_name, pinecone_api_key):
    """
    Convert raw chunked docs to LangChain Documents and upload to Pinecone via LangChain's vector store.

    Args:
        docs_after_split: List of dicts with 'text' and 'id' keys.
        embedding_model: LangChain-compatible embedding model.
        index_name: Pinecone index name.
        pinecone_api_key: Your Pinecone API key.
    """

                        
    # Convert to LangChain Document format
    documents = [
        Document(
            page_content=doc["text"],
            metadata={
                "source": doc["id"],
                "chapter" :doc["chapter"],
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        for doc in docs_after_split
    ]

    # Automatically embeds and uploads to Pinecone
    vector_store = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embedding_model,
        index_name=index_name,
        pinecone_api_key=pinecone_api_key,
    )

    print(f"✅ Uploaded {len(documents)} documents to Pinecone index '{index_name}'.")

    return vector_store