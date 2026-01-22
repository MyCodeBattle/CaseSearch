import sys
import os
import readline
from pathlib import Path

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
    
    print(f"Initializing Test RAG on {args.data_dir}...")
    
    # Initialize RAG
    rag = get_test_rag(data_dir=args.data_dir, allow_reset=False)

    
    def run_search(query_text):
        print(f"\n[Step 2-4] Running Search for: {query_text}")
        results = rag.search(query_text)
        
        print(f"\nFound {len(results)} matches:")
        for r in results:
            print(f"File: {r['filename']}")
            print(f"Score: {r['score']}")
            print(f"Reason: {r['reason']}")
            print("-" * 50)

    # 2. Search
    if args.query:
        run_search(args.query)
    else:
        # Interactive mode
        print("\n=== 进入交互式检索模式 ===")
        print("提示：输入 'q' 退出")
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
                print("\nExiting...")
                break
