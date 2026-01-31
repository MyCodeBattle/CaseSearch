import sys
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

project_root = Path(__file__).parent
sys.path.append(str(project_root))
load_dotenv()

from modules import legal_rag

def verify():
    rag = legal_rag.LegalRAG()
    collection = rag.abstract_collection
    
    count = collection.count()
    logger.info(f"Total items in abstract collection: {count}")
    
    if count > 0:
        results = collection.peek(limit=5)
        logger.info("\nSample Data (First 5):")
        ids = results['ids']
        metadatas = results['metadatas']
        documents = results['documents']
        
        for i in range(len(ids)):
            logger.info(f"\nID: {ids[i]}")
            logger.info(f"Metadata: {metadatas[i]}")
            logger.info(f"Content Start: {documents[i][:100]}...")

if __name__ == "__main__":
    verify()
