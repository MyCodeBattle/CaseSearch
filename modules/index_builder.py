import json
import os
import time
from pathlib import Path
from typing import List

from tqdm import tqdm

from .case_loader import get_available_types, load_cases_by_type


class IndexBuilderMixin:
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

            # Flush chunks for THIS file immediately to ensure separate logging
            if documents:
                # Set prefix for logging
                self.ef.log_prefix = f"text-{filename}"
                
                # Batch add to Chroma (max 10 chunks per request due to API limit)
                while documents:
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
                
                # Reset prefix
                self.ef.log_prefix = None

            if (case_idx + 1) % 100 == 0:
                pass
                # print(f"Processed {case_idx + 1}/{len(all_cases)} cases...")

        # No remaining chunks to add outside the loop because we flush per case

        print(f"Indexed {chunk_counter} chunks in Vector Store.")

        duration = time.time() - start_time
        print(f"Index built in {duration:.2f} seconds.")

    def build_abstract_index(self, jsonl_path: str, limit: int = None):
        """
        Builds the vector index for abstracts from a JSONL file.
        """
        if not os.path.exists(jsonl_path):
            print(f"JSONL file not found at {jsonl_path}")
            return

        print(f"Building abstract index from {jsonl_path}...")
        start_time = time.time()

        # 1. Build a map of ID -> Filepath from total_texts
        # IDs are the leading digits of filenames
        print("Scanning total_texts for source files...")
        id_to_path = {}
        # Assuming cases are in subdirectories of data_dir or directly in data_dir
        # We'll use os.walk to find all txt files

        # Determine root data directory (usually total_texts)
        # We need to look at where the cases are.
        # Based on config, data_dir is "total_texts".

        search_roots = [self.data_dir]

        for search_root in search_roots:
            if not search_root.exists():
                continue
            for root, dirs, files in os.walk(search_root):
                for file in files:
                    if file.endswith(".txt"):
                        # Extract ID: "1006_..." -> "1006"
                        parts = file.split('_')
                        if parts and parts[0].isdigit():
                            file_id = parts[0]
                            full_path = str(Path(root) / file)
                            id_to_path[file_id] = full_path

        print(f"Mapped {len(id_to_path)} source files by ID.")

        # Load existing IDs to avoid duplicates
        existing_ids = set()
        try:
            # Fetch all IDs in collection
            existing_data = self.abstract_collection.get()
            if existing_data and existing_data['ids']:
                existing_ids = set(existing_data['ids'])
            print(f"Found {len(existing_ids)} existing chunks in Abstract Vector Store.")
        except Exception as e:
            print(f"Error checking existing abstracts: {e}")

        documents = []
        metadatas = []
        ids = []

        count = 0
        skipped = 0

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        print(f"Found {len(lines)} abstracts to process.")
        
        if limit and limit > 0:
            lines = lines[:limit]
            print(f"Limiting to first {limit} abstracts.")


        for line in tqdm(lines, desc="Indexing Abstracts"):
            try:
                data = json.loads(line)
                # Extract relevant fields

                custom_id = data.get('custom_id', '')
                if not custom_id:
                    continue

                # Extract numeric ID from custom_id using regex or split
                # "request-1006" -> "1006"
                file_id = None
                if custom_id.startswith("request-") and "idx" not in custom_id:
                    # request-1006
                    file_id = custom_id.replace("request-", "")

                # Retrieve source path
                source_path = ""
                if file_id and file_id in id_to_path:
                    source_path = id_to_path[file_id]

                # EXTRACT FILENAME STEM AS DOC_ID
                # This ensures consistency with the main text index and allows existing retrieve logic to find the file.
                real_doc_id = custom_id  # fallback
                if source_path:
                    # /path/to/1006_Name.txt -> 1006_Name
                    real_doc_id = Path(source_path).stem

                # Check if ANY chunk for this document exists?
                # Using the correct ID format: real_doc_id + "_abstract_0"
                if f"{real_doc_id}_abstract_0" in existing_ids:
                    skipped += 1
                    continue

                # The content is in response.body.choices[0].message.content
                try:
                    content = data['response']['body']['choices'][0]['message']['content']
                except (KeyError, IndexError, TypeError):
                    continue

                if not content:
                    continue

                # Chunking logic for abstracts: chunk_size=100, overlap=20
                chunks = self._split_text(content, chunk_size=100, overlap=20)

                for i, chunk in enumerate(chunks):
                    # Use a unique ID for the vector store, but keep doc_id metadata consistent
                    chunk_id = f"{real_doc_id}_abstract_{i}"

                    documents.append(chunk)
                    metadatas.append({
                        "doc_id": real_doc_id,  # Crucial: Must match filename stem for retrieval
                        "source": "abstract",
                        "source_path": source_path,
                        "file_id": file_id if file_id else "",
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    })
                    ids.append(chunk_id)
                    count += 1

                # Batch upsert logic per ABSTRACT/FILE
                # We are inside the loop for ONE abstract (line). 
                # Chunks here belong to one real_doc_id.
                
                if documents:
                     # Set prefix for logging
                    self.ef.log_prefix = f"abstract-{real_doc_id}"
                    
                    while documents:
                        batch_docs = documents[:10]
                        batch_metas = metadatas[:10]
                        batch_ids = ids[:10]
                        
                        self.abstract_collection.upsert(
                            documents=batch_docs,
                            metadatas=batch_metas,
                            ids=batch_ids
                        )
                        documents = documents[10:]
                        metadatas = metadatas[10:]
                        ids = ids[10:]
                    
                    # Reset prefix
                    self.ef.log_prefix = None

            except json.JSONDecodeError:
                continue

        # No remaining chunks to add outside loop as we flush per abstract

        print(f"Indexed {count} abstracts in 'legal_abstracts' collection.")
        print(f"Abstract Index built in {time.time() - start_time:.2f} seconds.")
