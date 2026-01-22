import os
import shutil
from pathlib import Path
import json
import jieba
import time
import jieba
import time
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Set, Any, Tuple
from openai import OpenAI
import concurrent.futures
from tqdm import tqdm
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Import internal modules (assuming run from root or proper pythonpath)
try:
    from .case_loader import load_config, get_available_types, load_cases_by_type
    from .similarity_search import batch_cases_by_chars
except ImportError:
    # For testing when running directly
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from modules.case_loader import load_config, get_available_types, load_cases_by_type
    from modules.similarity_search import batch_cases_by_chars
    from modules.prompts import QUERY_EXPANSION_PROMPT, SIMILARITY_SEARCH_PROMPT
else:
    from .similarity_search import batch_cases_by_chars
    from .prompts import QUERY_EXPANSION_PROMPT, SIMILARITY_SEARCH_PROMPT

class LegalRAG:
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
                
                print(f"DEBUG: Embedding Type: {type(final_embeddings)}, Element Type: {type(final_embeddings[0])}, Inner Type: {type(final_embeddings[0][0])}")
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
        model_name = emb_config.get('model', 'text-embedding-3-small')
        
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
        print(f"DEBUG: Initialized Alibaba Embedding Function with model='{model_name}', dim=2048")
        
    def _split_text(self, text: str, chunk_size: int = 300, overlap: int = 100) -> List[str]:
        """
        Simple sliding window splitter
        """
        if not text:
            return []
            
        chunks = []
        if len(text) <= chunk_size:
            return [text]
            
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            if end >= len(text):
                break
            start += (chunk_size - overlap)
        return chunks

    def build_index(self, force_rebuild: bool = False):
        """
        Step 1: Data Slicing & Indexing
        """
        # Check if already indexed
        if not force_rebuild and self.collection.count() > 0:
             print("Index exists. Checking for new files to add...")
             # Do not return, continue to incremental logic
        else:
             print("Building index...")
        start_time = time.time()
        
        # 1. Load all cases
        all_cases = []
        types = get_available_types()
        print(f"Loading cases from types: {types}")
        
        for t in types:
            cases = load_cases_by_type(t)
            for c in cases:
                c['type'] = t
                all_cases.append(c)
        
        if not all_cases:
            print("No cases found to index. Exiting.")
            return

        print(f"Loaded {len(all_cases)} cases. processing...")
        
        # 2. Smart Indexing (Incremental)
        # Don't delete collection blindly. Check what exists.
        
        existing_doc_ids = set()
        try:
             # Fetch all metadata to find existing documents
             # Note: For very large collections, iterating batch-wise might be better, 
             # but strictly 'get' with include=['metadatas'] for all is easiest for now if manageable size.
             # Ideally we check existence case-by-case to avoid loading all metadata if huge.
             # However, making one query per file is also slow. 
             # Let's fetch all "doc_id"s. 
             
             # To avoid OOM on huge datasets, we'll scan in batches if needed, but let's assume it fits 
             # or check by filename availability?
             # For this task, getting all metadata is acceptable.
             existing_data = self.collection.get(include=['metadatas'])
             if existing_data and existing_data['metadatas']:
                 for m in existing_data['metadatas']:
                     if m:
                        existing_doc_ids.add(m.get('doc_id'))
             print(f"Found {len(existing_doc_ids)} existing documents in Vector Store.")
        except Exception as e:
            print(f"Error checking existing documents: {e}")

        documents = []
        metadatas = []
        ids = []
        
        documents = []
        metadatas = []
        ids = []
        
        chunk_counter = 0

        for case_idx, case in enumerate(tqdm(all_cases, desc="Indexing Cases", unit="case")):
            filename = case['filename']
            is_indexed = filename in existing_doc_ids
            
            # 2.1 Split text
            chunks = self._split_text(case['content'], chunk_size=300, overlap=100)
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{filename}_{i}"
                
                # Chroma Data -> ONLY if not already indexed
                if not is_indexed:
                    documents.append(chunk)
                    metadatas.append({
                        "doc_id": filename,
                        "filepath": str(case['filepath']),
                        "type": case['type'],
                        "chunk_index": i,
                        "parent_len": len(case['content'])
                    })
                    ids.append(chunk_id)
                    chunk_counter += 1
            
            if not is_indexed:
                # Batch add to Chroma (max 10 chunks per request due to API limit)
                while len(documents) >= 10:
                    batch_docs = documents[:10]
                    batch_metas = metadatas[:10]
                    batch_ids = ids[:10]
                    

                    self.collection.add(
                        documents=batch_docs,
                        metadatas=batch_metas,
                        ids=batch_ids
                    )
                    
                    # Remove processed
                    documents = documents[10:]
                    metadatas = metadatas[10:]
                    ids = ids[10:]
            elif (case_idx + 1) % 10 == 0:
                 # Just log occasionally for skipped
                 pass

            if (case_idx + 1) % 100 == 0:
                pass
                # print(f"Processed {case_idx + 1}/{len(all_cases)} cases...")

        # Add remaining chunks to Chroma
        # Add remaining chunks to Chroma
        while documents:
            batch_docs = documents[:10]
            batch_metas = metadatas[:10]
            batch_ids = ids[:10]
            

            self.collection.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids
            )
            
            documents = documents[10:]
            metadatas = metadatas[10:]
            ids = ids[10:]
            
        print(f"Indexed {chunk_counter} chunks in Vector Store.")
        
        duration = time.time() - start_time
        print(f"Index built in {duration:.2f} seconds.")

    def get_openai_client(self):
        return OpenAI(
            api_key=self.config['openai']['api_key'],
            base_url=self.config['openai']['base_url']
        )

    def query_expansion(self, user_query: str) -> Dict[str, Any]:
        """
        Step 2: Query Decomposition
        """
        # Load prompt from file (optional) or use inline
        # Using centralized prompt from prompts.py
        
        client = self.get_openai_client()
        prompt = QUERY_EXPANSION_PROMPT.format(user_query=user_query)
        
        try:
            # Use query_model for query expansion
            model = self.config['openai'].get('query_model') or self.config['openai'].get('model')
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个JSON输出助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            print(f"Query expansion failed: {e}")
            # Fallback to simple query
            return {
                "conditions": [{
                    "id": 1, 
                    "description": "Original Query", 
                    "keywords": list(jieba.cut(user_query)), 
                    "vector_query": user_query
                }], 
                "logic": "AND"
            }

    def retrieve(self, conditions: List[Dict], logic: str = "AND") -> Dict[str, str]:
        """
        Step 3: Dual-Route Retrieval & Logic Intersection
        """
        # DEBUG LOGGING START
        debug_file = "debug_retrieval_results.txt"
        with open(debug_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n{'='*40}\nRetrieval Start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Logic: {logic}\n")
            f.write(f"Conditions: {json.dumps(conditions, ensure_ascii=False)}\n")
        # DEBUG LOGGING END

        candidate_maps = []
        
        for condition in conditions:
            # 1. Vector Search
            vector_query = condition.get('vector_query', '')
            vector_docs_map = {}
            
            if vector_query:
                # Use query_texts
                print(f"Calling embedding model for query: '{vector_query}'...")
                
                # DEBUG: Manual embedding to inspect and bypass potential internal issue
                try:
                    manual_embeddings = self.ef([vector_query])
                    print(f"DEBUG: Manual embedding shape: {len(manual_embeddings)} x {len(manual_embeddings[0])}")
                    print(f"DEBUG: First element type: {type(manual_embeddings[0][0])}")
                except Exception as e:
                    print(f"DEBUG: Manual embedding failed: {e}")
                    raise e
                    
                # EXPAND: Retrieve top N chunks
                vector_search_top_k = self.config['search'].get('vector_search_top_k', 500)
                results = self.collection.query(
                    query_embeddings=manual_embeddings,
                    # query_texts=[vector_query], # Use embeddings directly
                    n_results=vector_search_top_k
                )
                
                # COLLAPSE & TRUNCATE: Keep top 100 unique documents
                seen_docs = set()
                
                if results['metadatas']:
                    # Results and distances are returned in order
                    metas = results['metadatas'][0]
                    dists = results['distances'][0] if results['distances'] else [None] * len(metas)
                    
                    for meta, dist in zip(metas, dists):
                        did = meta['doc_id']
                        if did not in seen_docs:
                            seen_docs.add(did)
                            
                            doc_type = meta.get('type', '')
                            if doc_type:
                                dynamic_path = self.data_dir / doc_type / f"{did}.txt"
                            else:
                                dynamic_path = self.data_dir / f"{did}.txt"
                                
                            vector_docs_map[did] = {
                                "path": str(dynamic_path),
                                "distance": dist
                            }
                        
                            candidate_limit = self.config['search'].get('candidate_limit', 500)
                            if len(vector_docs_map) >= candidate_limit:
                                break
            
            # BM25 Search removed as per configuration
            
            condition_candidates = vector_docs_map
            # Use only paths for the next step, but keep scores for logging? 
            # The current architecture expects Dict[str, str] (id->path) for candidate_maps.
            # So we separate them.
            
            candidate_maps.append({k: v["path"] for k, v in condition_candidates.items()})
            print(f"Condition '{condition.get('description')}' found {len(condition_candidates)} candidates.")

            # DEBUG LOGGING
            with open("debug_retrieval_results.txt", "a", encoding="utf-8") as f:
                f.write(f"\n--- Condition: {condition.get('description')} ---\n")
                f.write(f"Found {len(condition_candidates)} candidates (Top 500 max).\n")
                for i, (did, info) in enumerate(condition_candidates.items()):
                    dist_str = f" (Distance: {info['distance']:.4f})" if info['distance'] is not None else ""
                    f.write(f"{i+1}. {did}{dist_str}\n")
            # END DEBUG LOGGING

        if not candidate_maps:
            return {}

        # Apply Logic
        final_candidates = candidate_maps[0]
        if logic == "AND":
             for m in candidate_maps[1:]:
                 # Intersect keys
                 common_keys = final_candidates.keys() & m.keys()
                 final_candidates = {k: final_candidates[k] for k in common_keys}
                 
        elif logic == "OR":
             for m in candidate_maps[1:]:
                 final_candidates.update(m)
                 
        print(f"After {logic} logic, {len(final_candidates)} candidates remain.")
        
        # DEBUG LOGGING
        with open("debug_retrieval_results.txt", "a", encoding="utf-8") as f:
             f.write(f"\n--- Final Result (Logic: {logic}) ---\n")
             f.write(f"Remaining {len(final_candidates)} candidates:\n")
             for i, did in enumerate(final_candidates.keys()):
                 f.write(f"{i+1}. {did}\n")
             f.write(f"{'='*40}\n")
        # END DEBUG LOGGING

        return final_candidates

    def analyze_candidates(self, doc_id_map: Dict[str, str], user_query: str) -> List[Dict]:
        """
        Step 4: Full-Text Analysis (Batch)
        Matches logic in similarity_search.py
        """
        # Load full content
        candidates = []
        
        # Iterate provided map to find filepaths
        for did, filepath in doc_id_map.items():
             if filepath:
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        first_line = content.split('\n')[0].strip() if content else ""
                        candidates.append({
                            'filename': did,
                            'filepath': filepath,
                            'content': content, 
                            'char_count': len(content),
                            'header_info': first_line
                        })
                except Exception as e:
                    print(f"Failed to load file {filepath}: {e}")
                        
        if not candidates:
            return []
            
        # Match splitting logic from proper similarity search
        max_chars = self.config['search']['max_chars_per_request']
        batches = batch_cases_by_chars(candidates, max_chars)
        
        all_results = []
        print(f"Candidates split into {len(batches)} batches for analysis.")
        
        final_limit = self.config['search'].get('final_result_limit', 10)
        
        for batch in batches:
            # Pass final_limit to batch analysis to let LLM know the target count, 
            # though LLM sees only one batch at a time.
            batch_results = self._analyze_batch(user_query, batch, top_k=final_limit)
            all_results.extend(batch_results)
        
        # Sort by score
        all_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        # Global Truncation
        final_limit = self.config['search'].get('final_result_limit', 10)
        return all_results[:final_limit]

    def _analyze_batch(self, query: str, cases: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        Batch analysis using the prompt from similarity_search.py
        """
        # Step 4 uses independent analysis config
        analysis_config = self.config.get('analysis', {})
        api_key = analysis_config.get('api_key') or self.config['openai']['api_key']
        base_url = analysis_config.get('base_url') or self.config['openai']['base_url']
        model = analysis_config.get('model') or self.config['openai']['model']

        client = OpenAI(api_key=api_key, base_url=base_url)
        
        # Build Case Text
        cases_text = ""
        for i, case in enumerate(cases, 1):
            cases_text += f"\n\n===== 案例 {i}: {case['filename']} =====\n{case['content']}\n"
        
        final_prompt = SIMILARITY_SEARCH_PROMPT.format(
            query=query,
            cases_text=cases_text,
            top_k=min(top_k, len(cases))
        )

        try:
            # Use analysis model
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个JSON输出助手，只输出有效的JSON格式。"},
                    {"role": "user", "content": final_prompt}
                ],
                temperature=0.2, # Matching similarity_search.py
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content.strip()
            print(f"DEBUG: Batch Analysis Response:\n{content}\n")
            # Basic cleanup
            if content.startswith("```json"): content = content[7:]
            if content.startswith("```"): content = content[3:]
            if content.endswith("```"): content = content[:-3]
            
            result_json = json.loads(content.strip())
            api_results = result_json.get('results', [])
            
            # Map back to full case info
            case_map = {c['filename']: c for c in cases}
            final_results = []
            
            for res in api_results:
                fname = res.get('filename')
                if fname in case_map:
                    # Filter by min_score
                    if res.get('similarity_score', 0) >= self.config['search'].get('min_score', 60):
                         merged = case_map[fname].copy()
                         merged.update(res)
                         # rename score to match RAG output format expected by test?
                         # Test expects 'score' key. api returns 'similarity_score'.
                         merged['score'] = res.get('similarity_score') 
                         final_results.append(merged)
            
            return final_results

        except Exception as e:
            print(f"Batch analysis failed: {e}")
            return []

    def search(self, user_query: str, progress_callback=None):
        # Pipeline
        # 1. Index (assumed built)
        # 2. Query Decomposition
        if progress_callback:
            progress_callback("正在进行查询语义分解...")
        print(f"Step 2: Decomposing query: {user_query}")
        query_plan = self.query_expansion(user_query)
        print("Query Plan:", json.dumps(query_plan, ensure_ascii=False, indent=2))
        
        # 3. Retrieve
        if progress_callback:
            progress_callback("正在向量数据库中检索相似片段...")
        print("Step 3: Retrieving candidates...")
        candidates = self.retrieve(query_plan.get('conditions', []), logic=query_plan.get('logic', 'AND'))
        
        # 4. Verify
        if progress_callback:
            progress_callback(f"初筛找到 {len(candidates)} 个相关文档，正在使用大模型进行深度比对与验证...")
        print(f"Step 4: Verifying {len(candidates)} candidates with LLM (Analysis Model)...")
        final_results = self.analyze_candidates(candidates, user_query)
        
        if progress_callback:
            progress_callback("正在整理最终结果...")
        
        return final_results

if __name__ == "__main__":
    # Test script
    rag = LegalRAG()
    # rag.build_index() # Comment out to avoid rebuild every time
    
    query = "司法拍卖取得股份后违规减持"
    results = rag.search(query)
    
    print(f"\nFound {len(results)} matches:")
    for r in results:
        print(f"- [{r['score']}] {r['filename']}: {r['reason']}")


