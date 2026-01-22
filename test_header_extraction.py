
import sys
import os
sys.path.append(os.getcwd())
from modules.legal_rag import LegalRAG

print("Initializing LegalRAG...")
rag = LegalRAG()
# Mock the retrieval or search to avoid full costs if possible, but actually we need to test reading the file.
# We can just call verify directly if we have a doc id?
# Or just search a nonsense query that matches something by vector.

query = "内幕交易"
print(f"Searching for: {query}")
# We need to use force_rebuild=False (default)

try:
    results = rag.search(query)
    if results:
        first_result = results[0]
        print("\n--- First Result Keys ---")
        print(first_result.keys())
        print("\n--- Header Info ---")
        print(f"'{first_result.get('header_info')}'")
        
        if 'header_info' in first_result and first_result['header_info']:
            print("SUCCESS: header_info found.")
        else:
            print("FAILURE: header_info missing or empty.")
    else:
        print("No results found.")
except Exception as e:
    print(f"Error: {e}")
