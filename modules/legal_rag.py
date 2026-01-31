from pathlib import Path
from typing import Any, List

import json
import time
import uuid
import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from loguru import logger

# Load env vars
load_dotenv()

# Import internal modules (assuming run from root or proper pythonpath)
try:
    from .config_loader import load_config
    from .index_builder import IndexBuilderMixin
    from .rag_search import SearchMixin
except ImportError:
    # For testing when running directly
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from modules.config_loader import load_config
    from modules.index_builder import IndexBuilderMixin
    from modules.rag_search import SearchMixin

class LegalRAG(IndexBuilderMixin, SearchMixin):
    def __init__(self):
        self.config = load_config()
        self.root_dir = Path(__file__).parent.parent
        self.data_dir = self.root_dir / self.config['data_dir']
        self.chroma_db_path = str(self.data_dir / "chroma_db")
        
        # Initialize ChromaDB
        # Use simple persistent client
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_db_path)

        # Custom Embedding Function for Alibaba (supports dimensions)
        from chromadb import EmbeddingFunction, Documents, Embeddings

        class AlibabaEmbeddingFunction(EmbeddingFunction):
            def __init__(self, api_key, base_url, model_name="text-embedding-v4", dimensions=None):
                self.client = OpenAI(api_key=api_key, base_url=base_url)
                self.model_name = model_name
                self.dimensions = dimensions
                self.log_prefix = None # Context for logging filenames

            def __call__(self, input: Documents) -> Embeddings:
                if not input:
                    return []
                # Clean empty strings to avoid API errors
                valid_input = [text if text else " " for text in input]
                
                params = {
                    "model": self.model_name,
                    "input": valid_input
                }
                if self.dimensions:
                    params["dimensions"] = self.dimensions
                    
                resp = self.client.embeddings.create(**params)

                # Save intermediate result
                try:
                    output_dir = Path("intermediate_results")
                    output_dir.mkdir(exist_ok=True)
                    
                    filename = output_dir / f"{self.log_prefix}.json"
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(resp.model_dump_json(indent=2))
                except Exception as e:
                    logger.warning(f"Warning: Failed to save intermediate embedding response: {e}")
                
                # Deep conversion to ensure native python types
                final_embeddings = []
                for d in resp.data:
                    emb = d.embedding
                    # Check if it's numpy
                    if hasattr(emb, 'tolist'):
                        emb = emb.tolist()
                    
                    # Ensure every element is a recursive python float
                    # (OpenAI usually returns list of floats, but just in case)
                    if isinstance(emb, list):
                        emb = [float(x) for x in emb]
                    
                    final_embeddings.append(emb)
                
                logger.debug(f"DEBUG: Embedding Type: {type(final_embeddings)}, Element Type: {type(final_embeddings[0])}, Inner Type: {type(final_embeddings[0][0])}")
                return final_embeddings
            
            def embed_query(self, input: Any) -> List[float]:
                """
                Alias for single query embedding. Handles both str and list[str] (single item) just in case.
                """
                if isinstance(input, str):
                    input = [input]
                return self(input)[0]

            def embed_documents(self, input: List[str]) -> List[List[float]]:
                """
                Alias for document embedding
                """
                return self(input)

            @staticmethod
            def name():
                return "alibaba_embedding_function"
        
        # Prepare params
        emb_config = self.config.get('embedding', {})
        api_key = emb_config.get('api_key') or self.config['openai']['api_key']
        base_url = emb_config.get('base_url') or self.config['openai']['base_url']
        model_name = emb_config.get('model', 'error')
        
        self.ef = AlibabaEmbeddingFunction(
                api_key=api_key,
                base_url=base_url,
                model_name=model_name,
                dimensions=2048
            )
        
        self.collection = self.chroma_client.get_or_create_collection(
            name="legal_cases",
            embedding_function=self.ef
        )
        # New collection for abstracts
        self.abstract_collection = self.chroma_client.get_or_create_collection(
            name="legal_abstracts",
            embedding_function=self.ef
        )
        logger.debug(f"DEBUG: Initialized Alibaba Embedding Function with model='{model_name}', dim=2048")
        
if __name__ == "__main__":
    # Test script
    rag = LegalRAG()
    # rag.build_index() # Comment out to avoid rebuild every time
    
    query = "司法拍卖取得股份后违规减持"
    results = rag.search(query)
    
    logger.info(f"\nFound {len(results)} matches:")
    for r in results:
        logger.info(f"- [{r['score']}] {r['filename']}: {r['reason']}")

