import json
import time
from typing import Any, Dict, List

import jieba
from openai import OpenAI

from .prompts import QUERY_EXPANSION_PROMPT, SIMILARITY_SEARCH_PROMPT
from loguru import logger

def batch_cases_by_chars(cases: List[Dict], max_chars: int) -> List[List[Dict]]:
    """
    按字数阈值将案例分批
    
    Args:
        cases: 案例列表
        max_chars: 每批最大字数
    
    Returns:
        分批后的案例列表
    """
    batches = []
    current_batch = []
    current_chars = 0
    
    for case in cases:
        case_chars = case['char_count']
        
        # 如果当前案例加入后会超过阈值，且当前批次不为空，先保存当前批次
        if current_chars + case_chars > max_chars and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_chars = 0
        
        current_batch.append(case)
        current_chars += case_chars
    
    # 最后一批
    if current_batch:
        batches.append(current_batch)
    
    return batches



class SearchMixin:
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
                stream=False,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
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

    def retrieve_abstracts(self, user_query: str) -> Dict[str, str]:
        """
        Step 3.1: Abstract-based Retrieval (Top 300)
        Uses the full user query for retrieval.
        """
        logger.info(f"Step 3.1: Retrieving from Abstract Index using full query...")
        
        # Default to 300 documents
        target_doc_count = 300
        # Fetch more chunks to ensure we get enough unique docs (e.g. 3x)
        vector_search_k = target_doc_count * 3 

        vector_query = user_query
        candidate_map = {}

        if vector_query:
            try:
                # Embed query
                # Using the same embedding function as main index (assuming compatible)
                manual_embeddings = self.ef([vector_query])
                
                # Search abstract collection
                results = self.abstract_collection.query(
                    query_embeddings=manual_embeddings,
                    n_results=vector_search_k
                )
                
                # Process results
                seen_docs = set()
                if results['metadatas']:
                    metas = results['metadatas'][0]
                    dists = results['distances'][0] if results['distances'] else [None] * len(metas)

                    for meta, dist in zip(metas, dists):
                        # In abstract index, metadatas should have 'doc_id' which is the filename stem
                        did = meta.get('doc_id')
                        if not did:
                            continue
                            
                        if did not in seen_docs:
                            seen_docs.add(did)
                            
                            # Reconstruct path from doc_id or use stored source_path
                            # We need 'path' for the subsequent steps (reading content)
                            # Abstract metadata has 'source_path'
                            source_path = meta.get('source_path')
                            if not source_path:
                                # Fallback logic if source_path missing
                                # Assume standard location? Or skip.
                                pass

                            candidate_map[did] = {
                                "path": str(source_path) if source_path else "",
                                "distance": dist
                            }

                            if len(candidate_map) >= target_doc_count:
                                break
            except Exception as e:
                logger.error(f"Abstract search failed for query '{vector_query}': {e}")
        
        # In the previous logic we returned a Dict[str, str] (id -> path)
        # We should maintain that interface
        final_candidates = {k: v["path"] for k, v in candidate_map.items()}
        
        logger.info(f"Abstract search found {len(final_candidates)} candidates.")
        return final_candidates

    def retrieve(self, conditions: List[Dict], logic: str = "AND", filter_ids: List[str] = None) -> Dict[str, str]:
        """
        Step 3.2: Full-Text Retrieval (Existing Logic)
        Optionally filtered by candidate IDs
        """
        # DEBUG LOGGING START
        debug_file = "debug_retrieval_results.txt"
        with open(debug_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n{'='*40}\nFull Text Retrieval Start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Logic: {logic}\n")
            f.write(f"Filter IDs count: {len(filter_ids) if filter_ids else 'None'}\n")
        # DEBUG LOGGING END

        candidate_maps = []

        # Prepare filter
        chroma_filter = None
        if filter_ids:
            # Chroma matches "doc_id" in metadata
            # $in takes a list
            # WARNING: Large lists might be an issue. 300 should be safe.
            chroma_filter = {"doc_id": {"$in": filter_ids}}

        for condition in conditions:
            # 1. Vector Search
            vector_query = condition.get('vector_query', '')
            vector_docs_map = {}

            if vector_query:
                # Use query_texts
                logger.debug(f"Calling embedding model for query (Full Text): '{vector_query}'...")

                try:
                    manual_embeddings = self.ef([vector_query])
                except Exception as e:
                    logger.error(f"DEBUG: Manual embedding failed: {e}")
                    raise e

                # EXPAND: Retrieve top N chunks
                vector_search_top_k = self.config['search'].get('vector_search_top_k', 500)
                
                query_params = {
                    "query_embeddings": manual_embeddings,
                    "n_results": vector_search_top_k
                }
                if chroma_filter:
                    query_params["where"] = chroma_filter

                results = self.collection.query(**query_params)

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
            logger.info(f"Condition '{condition.get('description')}' found {len(condition_candidates)} candidates.")

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

        logger.info(f"After {logic} logic, {len(final_candidates)} candidates remain.")

        # DEBUG LOGGING
        with open("debug_retrieval_results.txt", "a", encoding="utf-8") as f:
            f.write(f"\n--- Final Result (Logic: {logic}) ---\n")
            f.write(f"Remaining {len(final_candidates)} candidates:\n")
            for i, did in enumerate(final_candidates.keys()):
                f.write(f"{i+1}. {did}\n")
            f.write(f"{'='*40}\n")
        # END DEBUG LOGGING

        return final_candidates

    def analyze_candidates(self, doc_id_map: Dict[str, str], user_query: str, progress_callback=None) -> List[Dict]:
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
                    logger.error(f"Failed to load file {filepath}: {e}")

        if not candidates:
            return []

        # Match splitting logic from proper similarity search
        max_chars = self.config['search']['max_chars_per_request']
        batches = batch_cases_by_chars(candidates, max_chars)

        all_results = []
        logger.info(f"Candidates split into {len(batches)} batches for analysis.")

        final_limit = self.config['search'].get('final_result_limit', 10)

        for batch in batches:
            # Pass final_limit to batch analysis to let LLM know the target count,
            # though LLM sees only one batch at a time.
            batch_results = self._analyze_batch(user_query, batch, top_k=final_limit, progress_callback=progress_callback)
            all_results.extend(batch_results)

        # Sort by score
        all_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)

        # Global Truncation
        final_limit = self.config['search'].get('final_result_limit', 10)
        return all_results[:final_limit]

    def _analyze_batch(self, query: str, cases: List[Dict], top_k: int = 10, progress_callback=None) -> List[Dict]:
        """
        Batch analysis using the prompt from similarity_search.py
        """
        # Step 4 uses independent analysis config
        analysis_config = self.config.get('analysis', {})
        api_key = analysis_config.get('api_key') or self.config['openai']['api_key']
        base_url = analysis_config.get('base_url') or self.config['openai']['base_url']
        model = analysis_config.get('model') or self.config['openai']['model']
        # client = OpenAI(api_key=api_key, base_url=base_url)
        client = OpenAI(api_key=api_key, base_url=base_url)

        # Build Case Text
        cases_text = ""
        for i, case in enumerate(cases, 1):
            cases_text += f"\n\n===== filename:{case['filename']} =====\n{case['content']}\n"

        final_prompt = SIMILARITY_SEARCH_PROMPT.format(
            query=query,
            cases_text=cases_text,
            top_k=min(top_k, len(cases))
        )

        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                # Use analysis model
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "你是一个JSON输出助手，只输出有效的JSON格式。"},
                        {"role": "user", "content": final_prompt}
                    ],
                    temperature=0,  # Matching similarity_search.py
                    stream=False,
                    response_format={"type": "json_object"}
                )

                content = response.choices[0].message.content.strip()
                logger.debug(f"DEBUG: Batch Analysis Response:\n{content}\n")
                # Basic cleanup
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]

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
                logger.error(f"Batch analysis failed (Attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    if progress_callback:
                        progress_callback("大模型请求失败，重试中……继续喝咖啡吧☕️~")
                    time.sleep(retry_delay)
                else:
                    return []

    def search(self, user_query: str, progress_callback=None):
        # Pipeline
        # 1. Index (assumed built)
        # 2. Query Decomposition
        if progress_callback:
            progress_callback("正在进行查询语义分解...")
        logger.info(f"Step 2: Decomposing query: {user_query}")
        query_plan = self.query_expansion(user_query)
        logger.debug(f"Query Plan: {json.dumps(query_plan, ensure_ascii=False, indent=2)}")

        # 3. Retrieve
        # 3. Retrieve
        if progress_callback:
            progress_callback("正在进行两阶段检索（摘要初筛 -> 全文精检）...")
        
        # Step 3.1: Abstract Retrieval
        # First, retrieve top 300 candidates based on detailed query
        logger.info("Step 3.1: Retrieving abstracts...")
        # Use simple query instead of decomposed conditions
        abstract_candidates = self.retrieve_abstracts(user_query)
        progress_callback("摘要初筛完成，正在进行全文精检...")
        # Step 3.2: Full Text Retrieval (Filtered by Abstract Candidates)
        # Using the existing retrieve logic, but passing the IDs from Step 3.1
        logger.info("Step 3.2: Retrieving full text candidates...")
        
        filter_ids = list(abstract_candidates.keys()) if abstract_candidates else None
        
        if not filter_ids:
            logger.warning("Warning: No candidates found in abstract search. Proceeding with full search or returning empty?")
            # For now, if no abstracts found, we might want to default to full search OR return empty.
            # let's proceed with full search (filter_ids=None) if that's safer, OR return empty.
            # Given "optimizing", let's assume if abstract fails, valid response is empty strictly? 
            # Or maybe fallback?
            # Let's fallback to full search if abstract yields nothing, or respect the emptiness.
            # If abstract search returns 0, filtering by [] will return 0.
            pass
        
        candidates = self.retrieve(
            query_plan.get('conditions', []), 
            logic=query_plan.get('logic', 'AND'),
            filter_ids=filter_ids
        )

        # 4. Verify
        if progress_callback:
            progress_callback(f"找到 {len(candidates)} 个相关文档，正在调用大模型进行深度比对与验证...该过程耗时较长，先喝杯咖啡吧！☕️")
        logger.info(f"Step 4: Verifying {len(candidates)} candidates with LLM (Analysis Model)...")
        final_results = self.analyze_candidates(candidates, user_query, progress_callback=progress_callback)

        if progress_callback:
            progress_callback("正在整理最终结果...")

        return final_results
