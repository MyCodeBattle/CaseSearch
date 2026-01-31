
import sys
import os
from loguru import logger
sys.path.append(os.getcwd())
from modules.legal_rag import LegalRAG

logger.info("Initializing LegalRAG...")
rag = LegalRAG()
# Mock the retrieval or search to avoid full costs if possible, but actually we need to test reading the file.
# We can just call verify directly if we have a doc id?
# Or just search a nonsense query that matches something by vector.

query = "内幕交易"
logger.info(f"Searching for: {query}")
# We need to use force_rebuild=False (default)

try:
    results = rag.search(query)
    if results:
        first_result = results[0]
        logger.info("\n--- First Result Keys ---")
        logger.info(first_result.keys())
        logger.info("\n--- Header Info ---")
        logger.info(f"'{first_result.get('header_info')}'")
        
        if 'header_info' in first_result and first_result['header_info']:
            logger.info("SUCCESS: header_info found.")
        else:
            logger.error("FAILURE: header_info missing or empty.")
    else:
        logger.warning("No results found.")
except Exception as e:
    logger.error(f"Error: {e}")
