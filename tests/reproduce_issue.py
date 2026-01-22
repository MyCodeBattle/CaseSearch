
import sys
import os
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from modules.legal_rag import LegalRAG
from tests.rag_test_utils import setup_test_environment

def check_case_1615():
    # Setup env to use total_texts
    setup_test_environment("total_texts")
    rag = LegalRAG()
    target_file = "1615_海陆重工_中国证券监督管理委员会江苏监管局行政处罚决定书徐元生.txt"
    
    print(f"RAG Data Dir: {rag.data_dir}")
    print(f"RAG DB Path: {rag.chroma_db_path}")
    
    # List all doc_ids to see what's there
    print("Fetching all doc_ids from metadata...")
    all_data = rag.collection.get(include=['metadatas'])
    found_any = False
    
    if all_data['metadatas']:
        for m in all_data['metadatas']:
             if m and 'doc_id' in m:
                 did = m['doc_id']
                 if "1615" in did:
                     print(f"  Found similar doc_id: {did}")
                     found_any = True
    
    if not found_any:
        print("❌ No document containing '1615' found in the entire index.")
        return

    # Check Query Expansion
    query = "有没有签订控制权变更备忘录没披露,被处罚的案例?"
    print(f"\nRunning Query Expansion for: {query}")
    plan = rag.query_expansion(query)
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    
    # Check Retrieval
    print("\nRunning Retrieval...")
    candidates = rag.retrieve(plan.get('conditions', []), logic=plan.get('logic', 'AND'))
    
    found = False
    for did in candidates.keys():
        if "1615" in did and "海陆重工" in did:
            found = True
            print(f"✅ Document found in candidate list! Path: {candidates[did]}")
            break
            
    if not found:
        print("❌ Document NOT found in candidate list.")
        
        # Debug: Check similarity score for this specific doc against the queries
        print("\nChecking vector similarity for specific document...")
        doc_embedding_result = rag.collection.get(
            where={"doc_id": "1615_海陆重工_中国证券监督管理委员会江苏监管局行政处罚决定书徐元生"},
            include=["embeddings", "documents"]
        )
        
        if doc_embedding_result['embeddings']:
            # We have chunks. Let's check the first chunk against the query.
            # We need to manually call embedding function
            for condition in plan.get('conditions', []):
                q = condition.get('vector_query')
                print(f"Query: {q}")
                # Analyze top results for this query
                res = rag.collection.query(
                    query_texts=[q],
                    n_results=10,
                    where={"doc_id": "1615_海陆重工_中国证券监督管理委员会江苏监管局行政处罚决定书徐元生"} # Forced filter
                )
                print(f"  Similarity to 1615 chunks (lower distance is better for L2, higher score for cosine - verify metric):")
                # Chroma default is L2 distance? Or did we set cosine? 
                # Checking retrieval output might give a clue, usually it returns distance.
                # If using custom embedding function, check config.
                
                if res['distances']:
                    print(f"  Distances: {res['distances'][0]}")
                else:
                    print("  No match found even with filter?")

if __name__ == "__main__":
    check_case_1615()
