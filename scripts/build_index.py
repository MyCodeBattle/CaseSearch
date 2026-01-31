import sys
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

load_dotenv()

import modules.case_loader
import modules.config_loader

def main():
    parser = argparse.ArgumentParser(description="Build ChromaDB index for Legal RAG")
    parser.add_argument("--data-dir", type=str, help="Override data directory (default: read from config.yaml)")
    parser.add_argument("--force", action="store_true", help="Force rebuild of the index")
    parser.add_argument("--abstracts", action="store_true", help="Build index for abstracts")
    parser.add_argument("--limit", type=int, help="Limit number of items to process (for testing)")
    
    args = parser.parse_args()
    
    target_data_dir = args.data_dir
    
    # If no CLI arg, check config
    if not target_data_dir:
        temp_config = modules.config_loader.load_config()
        target_data_dir = temp_config.get('build', {}).get('data_dir')

    # Patch config if target_data_dir is determined
    if target_data_dir:
        logger.info(f"Target data directory: {target_data_dir}")
        original_load_config = modules.config_loader.load_config
        
        def patched_load_config():
            config = original_load_config()
            config['data_dir'] = target_data_dir
            return config
            
        modules.config_loader.load_config = patched_load_config
        
        # Also patch legal_rag imports
        from modules import legal_rag
        legal_rag.load_config = patched_load_config
    else:
        # Fallback to default config (which uses 'data' or whatever is in root data_dir)
        logger.info("Using default data directory from config.")
        from modules import legal_rag

    logger.info("Initializing LegalRAG...")
    try:
        rag = legal_rag.LegalRAG()
        
        # If abstracts flag is set, only build abstract index
        if args.abstracts:
            logger.info("\nStarting Abstract Index Build...")
            jsonl_path = rag.data_dir / "abstract" / "results.jsonl"
            rag.build_abstract_index(str(jsonl_path), limit=args.limit)
            logger.info("\nAbstract Build Complete.")
            return

        logger.info("\nStarting Index Build...")
        if args.force:
            logger.info("Force rebuild enabled.")
        
        rag.build_index(force_rebuild=args.force)
        logger.info("\nBuild Complete.")
        
    except Exception as e:
        logger.error(f"\nError during build: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
