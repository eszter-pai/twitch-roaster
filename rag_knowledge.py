"""
RAG Knowledge Base for Witcher 3 lore using ChromaDB and EmbeddingGemma.
Chunks the_witcher_series.txt and stores embeddings for retrieval.
"""

import chromadb
from chromadb.utils import embedding_functions
import requests
import re
from pathlib import Path
from typing import List, Dict
import os
from dotenv import load_dotenv

load_dotenv()

# Get Ollama URL from environment variable
ollama_url = os.getenv('MODEL_API_URL_EMBED')
embedding_model_name = "embeddinggemma:300m"
witcher_text_file = "data/the_witcher_series.txt"


class OllamaEmbeddingFunction(embedding_functions.EmbeddingFunction):
    """Custom embedding function using Ollama's HTTP API."""
    
    def __init__(self, model_name: str, ollama_url: str):
        """
        Initialize Ollama embedding function.
        
        Args:
            model_name: Name of the Ollama embedding model 
                       (default: nomic-embed-text)
                       Other options: mxbai-embed-large, all-minilm
            ollama_url: URL of Ollama API (default: http://localhost:11434)
        """
        self.model_name = embedding_model_name
        self.embeddings_endpoint = ollama_url
        print(f"Initializing Ollama embedding function with model: {embedding_model_name}")
        print(f"Ollama API: {ollama_url}")
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using Ollama HTTP API.
        
        Args:
            input: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in input:
            try:
                response = requests.post(
                    self.embeddings_endpoint,
                    json={
                        "model": self.model_name,
                        "prompt": text
                    }
                )
                response.raise_for_status()
                response_data = response.json()
                
                # Handle different response formats
                if 'embedding' in response_data:
                    embedding = response_data['embedding']
                elif 'embeddings' in response_data:
                    embedding = response_data['embeddings']
                else:
                    print(f"Unexpected response format: {list(response_data.keys())}")
                    print(f"Full response: {response_data}")
                    # Return zero vector on error
                    embeddings.append([0.0] * 768)
                    continue
                    
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error generating embedding for text: {text[:50]}... Error: {e}")
                print(f"Response status: {response.status_code if 'response' in locals() else 'N/A'}")
                # Return zero vector on error
                embeddings.append([0.0] * 768)  # Default dimension
        
        return embeddings


def chunk_text_by_paragraphs(text: str, min_length: int = 100, max_length: int = 1000) -> List[str]:
    """
    Chunk text into paragraphs with size constraints.
    
    Args:
        text: Full text to chunk
        min_length: Minimum characters per chunk
        max_length: Maximum characters per chunk
        
    Returns:
        List of text chunks
    """
    # Split by double newlines (paragraphs)
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para or len(para) < 20:  # Skip very short or empty paragraphs
            continue
        
        # If adding this paragraph would exceed max_length, save current chunk
        if current_chunk and len(current_chunk) + len(para) > max_length:
            if len(current_chunk) >= min_length:
                chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            # Add to current chunk
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
    
    # Add final chunk
    if current_chunk and len(current_chunk) >= min_length:
        chunks.append(current_chunk.strip())
    
    return chunks


def load_and_chunk_witcher_text(file_path: str) -> List[str]:
    """
    Load and chunk the Witcher series text file.
    
    Args:
        file_path: Path to the_witcher_series.txt (defaults to global witcher_text_file)
        
    Returns:
        List of text chunks
    """
    if file_path is None:
        file_path = witcher_text_file
    print(f"Loading text from {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Loaded {len(text)} characters")
    print("Chunking text into paragraphs...")
    
    chunks = chunk_text_by_paragraphs(text)
    
    print(f"Created {len(chunks)} chunks")
    print(f"Average chunk size: {sum(len(c) for c in chunks) / len(chunks):.0f} characters")
    
    return chunks


def create_witcher_knowledge_base(
    db_path: str = "./chroma_db",
    collection_name: str = "witcher_lore",
    text_file: str = None,
    force_recreate: bool = False
) -> chromadb.Collection:
    """
    Create or load ChromaDB collection with Witcher lore embeddings.
    
    Args:
        db_path: Path to ChromaDB persistent storage
        collection_name: Name of the collection
        text_file: Path to the_witcher_series.txt (defaults to global witcher_text_file)
        force_recreate: If True, delete and recreate the collection
        
    Returns:
        ChromaDB collection
    """
    print("\n" + "="*50)
    print("Creating Witcher Knowledge Base with ChromaDB")
    print("="*50 + "\n")
    
    # Initialize ChromaDB client with persistent storage
    client = chromadb.PersistentClient(path=db_path)
    
    # Create custom embedding function using Ollama HTTP API
    embedding_function = OllamaEmbeddingFunction(
        model_name=embedding_model_name,
        ollama_url=ollama_url
    )
    
    # Check if collection exists
    existing_collections = [col.name for col in client.list_collections()]
    
    if collection_name in existing_collections:
        if force_recreate:
            print(f"Deleting existing collection: {collection_name}")
            client.delete_collection(name=collection_name)
        else:
            print(f"Loading existing collection: {collection_name}")
            collection = client.get_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            print(f"Collection has {collection.count()} documents")
            return collection
    
    # Create new collection
    print(f"Creating new collection: {collection_name}")
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata={"description": "The Witcher series lore and story content"}
    )
    
    # Load and chunk text
    if text_file is None:
        text_file = witcher_text_file
    chunks = load_and_chunk_witcher_text(text_file)
    
    # Prepare documents for ChromaDB
    print("\nGenerating embeddings and storing in ChromaDB...")
    print("(This may take a while depending on the number of chunks)")
    
    # Process in batches to avoid memory issues
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_ids = [f"chunk_{j}" for j in range(i, i+len(batch_chunks))]
        batch_metadatas = [
            {
                "chunk_index": j,
                "chunk_length": len(chunk),
                "source": text_file
            }
            for j, chunk in zip(range(i, i+len(batch_chunks)), batch_chunks)
        ]
        
        print(f"Processing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} "
              f"(chunks {i}-{i+len(batch_chunks)-1})")
        
        try:
            collection.add(
                documents=batch_chunks,
                ids=batch_ids,
                metadatas=batch_metadatas
            )
        except Exception as e:
            print(f"Error adding batch: {e}")
            continue
    
    print(f"\n✅ Successfully created knowledge base with {collection.count()} documents")
    print(f"📁 Stored at: {db_path}")
    
    return collection


def query_witcher_knowledge(
    query: str,
    collection: chromadb.Collection = None,
    n_results: int = 5,
    db_path: str = "./chroma_db",
    collection_name: str = "witcher_lore"
) -> Dict:
    """
    Query the Witcher knowledge base for relevant context.
    
    Args:
        query: Query text to search for
        collection: Existing ChromaDB collection (optional)
        n_results: Number of results to return
        db_path: Path to ChromaDB storage (if collection not provided)
        collection_name: Collection name (if collection not provided)
        
    Returns:
        Dictionary with documents, distances, and metadatas
    """
    if collection is None:
        # Load collection
        client = chromadb.PersistentClient(path=db_path)
        embedding_function = OllamaEmbeddingFunction(
            model_name=embedding_model_name,
            ollama_url=ollama_url
        )
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
    
    # Query the collection
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    return {
        'documents': results['documents'][0] if results['documents'] else [],
        'distances': results['distances'][0] if results['distances'] else [],
        'metadatas': results['metadatas'][0] if results['metadatas'] else []
    }


def format_retrieved_context(results: Dict, max_chunks: int = 3) -> str:
    """
    Format retrieved context for LLM prompt.
    
    Args:
        results: Results from query_witcher_knowledge
        max_chunks: Maximum number of chunks to include
        
    Returns:
        Formatted string with retrieved context
    """
    if not results['documents']:
        return ""
    
    context_parts = []
    for i, (doc, distance) in enumerate(zip(results['documents'][:max_chunks], 
                                            results['distances'][:max_chunks])):
        # Truncate very long chunks
        doc_preview = doc[:500] + "..." if len(doc) > 500 else doc
        context_parts.append(f"[Excerpt {i+1}, relevance: {1-distance:.2f}]\n{doc_preview}")
    
    formatted = "\n\n".join(context_parts)
    return f"RETRIEVED WITCHER LORE CONTEXT:\n{formatted}"


def main():
    """Main function to create and test the knowledge base."""
    import sys
    
    # Create knowledge base
    collection = create_witcher_knowledge_base(force_recreate=False)
    
    # Test queries
    print("\n" + "="*50)
    print("Testing Knowledge Base with Sample Queries")
    print("="*50 + "\n")
    
    test_queries = [
        "What is racism in the Witcher world?",
        "Tell me about elves and dwarves",
        "What are witchers?",
        "Nilfgaard empire politics"
    ]
    
    for query in test_queries:
        print(f"Query: {query}")
        print("-" * 50)
        
        results = query_witcher_knowledge(query, collection=collection, n_results=3)
        
        if results['documents']:
            print(f"Found {len(results['documents'])} relevant chunks:\n")
            for i, (doc, dist) in enumerate(zip(results['documents'][:2], 
                                                results['distances'][:2])):
                print(f"Result {i+1} (distance: {dist:.3f}):")
                print(doc[:300] + "...\n")
        else:
            print("No results found.\n")
        
        print()


if __name__ == "__main__":
    main()
