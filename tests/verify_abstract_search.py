
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from modules.legal_rag import LegalRAG

def test_abstract_search():
    print("Initializing LegalRAG...")
    try:
        rag = LegalRAG()
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return

    # Check abstract collection count
    count = rag.abstract_collection.count()
    print(f"Abstract Collection Count: {count}")
    
    if count == 0:
        print("WARNING: Abstract collection is empty. Please run 'python scripts/build_index.py --abstracts' first.")
        return

    query = "司法拍卖 股份 减持"
    print(f"\nPerforming search for: '{query}'")
    
    # Run search (which now includes abstract step)
    results = rag.search(query)
    
    print(f"\nFinal Results Count: {len(results)}")
    for r in results:
        print(f"- {r.get('filename')} (Score: {r.get('score')})")

if __name__ == "__main__":
    test_abstract_search()
