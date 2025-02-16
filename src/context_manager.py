import os
from pathlib import Path
from typing import List, Optional, Dict

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import tiktoken
from src.utils.logger import log

class ContextManager:
    def __init__(self, context_dir: str, config: Optional[Dict] = None):
        """Initialize the context manager with config settings"""
        log.info("🚀 Initializing Context Manager")
        self.context_dir = Path(context_dir)
        self.config = config or {}
        
        # Get vector db settings
        db_config = self.config.get("vector_db", {})
        self.db_dir = Path(db_config.get("persist_directory", "./chroma_db"))
        log.info(f"Using ChromaDB directory: {self.db_dir}")
        
        # Initialize ChromaDB with persistence
        log.info("Connecting to ChromaDB...")
        self.client = chromadb.PersistentClient(path=str(self.db_dir))
        
        # Use OpenAI embeddings as specified in config
        log.info("Setting up OpenAI embeddings...")
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=db_config.get("embedding_model", "text-embedding-ada-002")
        )
        
        # Get or create collection
        collection_name = db_config.get("index_name", "context_collection")
        log.info(f"Initializing collection: {collection_name}")
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

        # Initialize tokenizer
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        log.success("Context Manager initialized successfully")

    def chunk_text(self, text: str, chunk_size: int = 600, overlap_percentage: float = 0.15) -> List[str]:
        """Split text into chunks of approximately chunk_size tokens with overlap.
        
        Args:
            text: Text to split into chunks
            chunk_size: Target size of each chunk in tokens
            overlap_percentage: Percentage of overlap between chunks (0.0 to 1.0)
            
        Returns:
            List of text chunks with overlap
        """
        chunk_size = self.config.get("chunk_size", 600)
        overlap_percentage = self.config.get("chunk_overlap_percentage", 0.15)
        # Encode the text into tokens
        tokens = self.tokenizer.encode(text)
        
        if not tokens:
            log.warning("Empty text provided for chunking")
            return []
        
        overlap_size = int(chunk_size * overlap_percentage)
        stride = chunk_size - overlap_size
        
        chunks = []
        for i in range(0, len(tokens), stride):
            # Take chunk_size tokens, but don't exceed the array bounds
            chunk_tokens = tokens[i:min(i + chunk_size, len(tokens))]
            # Decode tokens back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        log.debug(f"Split text into {len(chunks)} chunks (size={chunk_size}, overlap={overlap_percentage})")
        return chunks

    def index_documents(self) -> None:
        """Index all markdown files in the context directory."""
        log.info("Starting document indexing")
        md_files = list(self.context_dir.glob('**/*.md'))
        
        if not md_files:
            log.warning(f"No markdown files found in {self.context_dir}")
            return
            
        log.info(f"Found {len(md_files)} markdown files to process")
        
        for file_path in md_files:
            try:
                log.info(f"Processing: {file_path.relative_to(self.context_dir)}")
                
                # Read the markdown file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Chunk the content
                chunks = self.chunk_text(content)
                log.info(f"Generated {len(chunks)} chunks from file")
                
                # Create IDs for each chunk
                chunk_ids = [f"{file_path.stem}_{i}" for i in range(len(chunks))]
                
                # Add metadata for each chunk
                metadatas = [{
                    "source": str(file_path),
                    "chunk_index": i
                } for i in range(len(chunks))]
                
                # Add to collection
                self.collection.add(
                    documents=chunks,
                    ids=chunk_ids,
                    metadatas=metadatas
                )
                log.success(f"Successfully indexed: {file_path.name}")
                
            except Exception as e:
                log.error(f"Failed to process {file_path.name}: {str(e)}")
                continue
        
        log.success("Document indexing completed")

    def query_context(self, query: str, n_results: int = 3) -> List[str]:
        """Query the vector database for relevant context chunks."""
        log.info(f"Querying context: '{query[:50]}...' (n_results={n_results})")
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            log.debug(f"Found {len(results['documents'][0])} matching chunks")
            
            # Print out the found chunks
            for i, chunk in enumerate(results['documents'][0]):
                log.info(f"\nChunk {i+1}:")
                log.info("=" * 40)
                log.info(chunk)
                log.info("=" * 40)
                
            return results['documents'][0]  # Return just the documents
        except Exception as e:
            log.error(f"Context query failed: {str(e)}")
            return []

    def reindex(self) -> None:
        """Clear the existing index and reindex all documents."""
        log.info("Starting reindexing process")
        
        try:
            log.info("Clearing existing index...")
            self.collection.delete(where={})
            
            log.info("Reindexing documents...")
            self.index_documents()
            
            log.success("Reindexing completed successfully")
            
        except Exception as e:
            log.error(f"Reindexing failed: {str(e)}")
            raise 