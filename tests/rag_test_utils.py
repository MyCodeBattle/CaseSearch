import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

load_dotenv()

import modules.case_loader
import modules.config_loader
from modules import legal_rag
from chromadb.utils import embedding_functions
import time
import numpy as np
import chromadb

# Global config for test data directory
# Default to sample_100_texts, but can be overridden
TARGET_DATA_DIR = "sample_100_texts"

def setup_test_environment(data_dir=None):
    """
    Sets up the test environment by patching load_config to use the specified data directory.
    """
    global TARGET_DATA_DIR
    if data_dir:
        TARGET_DATA_DIR = data_dir

    original_load_config = modules.config_loader.load_config

    def mocked_load_config():
        config = original_load_config()
        config['data_dir'] = TARGET_DATA_DIR
        # Ensure ChromaDB path is separate/test-specific
        if 'embedding' not in config:
            config['embedding'] = {}
            # Try to use env vars if present for test
            config['embedding']['api_key'] = os.environ.get('EMBEDDING_API_KEY') or os.environ.get('OPENAI_API_KEY')
            config['embedding']['base_url'] = os.environ.get('EMBEDDING_BASE_URL') or os.environ.get('OPENAI_BASE_URL')
            config['embedding']['model'] = os.environ.get('EMBEDDING_MODEL') or "text-embedding-3-small"
        return config

    # Apply patches
    modules.config_loader.load_config = mocked_load_config
    legal_rag.load_config = mocked_load_config
    
    return mocked_load_config

class VerboseOpenAIEmbeddingFunction(embedding_functions.OpenAIEmbeddingFunction):
    def __call__(self, input: list) -> list:
        # Start time
        t0 = time.time()
        print(f"\n[VerboseEF] Requesting embeddings for {len(input)} documents from Embedding API...")
        
        if not input:
            return []

        # Prepare embedding parameters
        embedding_params = {
            "model": self.model_name,
            "input": input,
        }

        # FORCE dimensions if set
        if self.dimensions is not None:
            embedding_params["dimensions"] = self.dimensions

        # Get embeddings
        response = self.client.embeddings.create(**embedding_params)

        # Extract embeddings from response
        results = [np.array(data.embedding, dtype=np.float32) for data in response.data]

        duration = time.time() - t0
        print(f"[VerboseEF] Received {len(results)} vectors in {duration:.2f}s. Vector Dim: {len(results[0])}")
        return results

class TestLegalRAG(legal_rag.LegalRAG):
    def __init__(self, allow_reset=False):
        super().__init__()
        # Override paths for test
        # Use filename-safe data_dir for db path
        safe_name = self.config['data_dir'].replace("/", "_").replace("\\", "_")
        
        # We want to keep using that structure but maybe customize the DB folder name to be sure?
        # Let's override to be explicit about "chroma_db_test" inside that dir.
        self.chroma_db_path = str(self.data_dir / "chroma_db")

        
        print(f"Test RAG paths:\nData: {self.data_dir}\nDB: {self.chroma_db_path}")
        
        # Re-init client with new path
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_db_path)
        
        # Consistent with main class
        emb_config = self.config.get('embedding', {})
        api_key = emb_config.get('api_key') or self.config['openai']['api_key']
        base_url = emb_config.get('base_url') or self.config['openai']['base_url']
        model_name = emb_config.get('model', 'text-embedding-3-small')

        # Use Verbose Wrapper
        openai_ef = VerboseOpenAIEmbeddingFunction(
                api_key=api_key,
                api_base=base_url,
                model_name=model_name,
                dimensions=2048
            )
        self.ef = openai_ef
        
        try:
            self.collection = self.chroma_client.get_or_create_collection(
                name="legal_cases",
                embedding_function=openai_ef
            )
        except ValueError as e:
            if "conflict" in str(e) and allow_reset:
                print("Embedding function conflict detected and allow_reset=True. Deleting existing collection...")
                self.chroma_client.delete_collection("legal_cases")
                self.collection = self.chroma_client.create_collection(
                    name="legal_cases",
                    embedding_function=openai_ef
                )
            elif "conflict" in str(e):
                print("\nWARNING: Embedding function metadata conflict (persisted vs new).")
                print("Forcing load of collection with new embedding function for verification.")
                self.collection = self.chroma_client.get_collection(name="legal_cases")
                self.collection._embedding_function = openai_ef
            else:
                raise e


def get_test_rag(data_dir="sample_100_texts", allow_reset=False):
    """
    Initializes and returns a TestLegalRAG instance with the specified configuration.
    """
    setup_test_environment(data_dir)
    return TestLegalRAG(allow_reset=allow_reset)
