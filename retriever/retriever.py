import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
import uuid
from collections import deque
from transformers import AutoTokenizer

class Retriever:
    def __init__(self, tokenizer, context_limit: int, chunk_limit = 200):
        """Initialize ChromaDB client with persistence"""
        self.client = chromadb.Client(Settings(
            is_persistent=False
        ))
        self.collection = self.client.get_or_create_collection("documents")
        self.tokenizer = tokenizer
        self.context_limit = context_limit
        self.chunk_limit = chunk_limit
        self.contextwindow = deque()

    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]] = None):
        # Generate IDs if not provided in metadata
        ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        
        # Add documents to collection
        self.collection.add(
            documents=documents,
            ids=ids
        )
        
    def add_to_contextwindow(self, new_text: str):
        # Split text into chunks if longer than chunk limit
        tokens = self.tokenizer.encode(new_text)
        if len(tokens) > self.chunk_limit:
            # Split into smaller chunks
            chunks = []
            for i in range(0, len(tokens), self.chunk_limit):
                chunk_tokens = tokens[i:i+self.chunk_limit]
                chunk_text = self.tokenizer.decode(chunk_tokens)
                chunks.append(chunk_text)
            
            # Add each chunk to context window
            for chunk in chunks:
                self.contextwindow.append(chunk)
        else:
            self.contextwindow.append(new_text)
        
        # Calculate total tokens in context window
        total_tokens = 0
        for text in self.contextwindow:
            tokens = self.tokenizer.encode(text)
            total_tokens += len(tokens)
            
        # If total tokens exceed limit, remove oldest items and add to ChromaDB
        while total_tokens > self.context_limit:
            oldest_text = self.contextwindow.popleft()
            self.add_documents([oldest_text])
            total_tokens -= len(self.tokenizer.encode(oldest_text))            

    def retrieve(self, query: str, n_results: int = 2) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )        
        retrieved_docs = [text for text in results['documents'][0]]
            
        return retrieved_docs    

def test_retriever():
    tokenizer = AutoTokenizer.from_pretrained(
        'lmsys/vicuna-13b-v1.3',
        trust_remote_code=True,
    )
    
    # Initialize retriever
    retriever = Retriever(tokenizer, context_limit=2000)
    
    # Test context window
    retriever.add_to_contextwindow("New context text")
    assert len(retriever.contextwindow) == 1
    
    # Test context window overflow
    long_text = "Long text 0 " * 1000  # Create text that exceeds chunk limit
    retriever.add_to_contextwindow(long_text)
    long_text2 = "My password is 898989" + "Long text 1 " * 1000  # Create text that exceeds chunk limit    
    retriever.add_to_contextwindow(long_text2)
    
    # Test retrieval
    query = "What is my secret?"
    results = retriever.retrieve(query, n_results=2)
    assert len(results) == 2
    assert "My password is 898989" in results[0]
    
    print("All tests passed!")
