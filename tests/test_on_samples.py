import sys
import os
import readline
from pathlib import Path
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from tests.rag_test_utils import get_test_rag

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test RAG on sample data")
    parser.add_argument("--data-dir", type=str, default="total_texts", help="Data directory name (relative to project root)")
    # Removed --build as it is now in build_chroma_db.py
    parser.add_argument("--query", type=str, help="Run search with this query", default=None)
    
    args = parser.parse_args()
    
    logger.info(f"Initializing Test RAG on {args.data_dir}...")
    
    # Initialize RAG
    rag = get_test_rag(data_dir=args.data_dir, allow_reset=False)

    
    def run_search(query_text):
        logger.info(f"\n[Step 2-4] Running Search for: {query_text}")
        results = rag.search(query_text)
        
        logger.info(f"\nFound {len(results)} matches:")
        for r in results:
            logger.info(f"File: {r['filename']}")
            logger.info(f"Score: {r['score']}")
            logger.info(f"Reason: {r['reason']}")
            logger.info("-" * 50)

    # 2. Search
    if args.query:
        run_search(args.query)
    else:
        # Interactive mode
        logger.info("\n=== 进入交互式检索模式 ===")
        logger.info("提示：输入 'q' 退出")
        while True:
            try:
                prompt = f"\n请输入检索语句 (输入 'q' 退出): "
                user_input = input(prompt).strip()
                
                if user_input.lower() in ('q', 'quit', 'exit'):
                    break
                

                    
                if not user_input:
                    continue
                run_search(user_input)
            except KeyboardInterrupt:
                logger.info("\nExiting...")
                break
