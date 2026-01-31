
import sys
import os
import unittest
from unittest.mock import MagicMock, patch
from loguru import logger

sys.path.append(os.getcwd())
from modules.legal_rag import LegalRAG

class TestHeaderExtraction(unittest.TestCase):
    def setUp(self):
        # Mock init to avoid loading embeddings/chroma
        with patch('modules.legal_rag.load_config', return_value={'data_dir': 'data', 'search': {'max_chars_per_request': 10000}}):
             with patch('modules.legal_rag.chromadb.PersistentClient'):
                 with patch('modules.legal_rag.LegalRAG.get_openai_client'):
                     self.rag = LegalRAG()
                     # Mock EF
                     self.rag.ef = MagicMock()
                     self.rag.collection = MagicMock()

    def test_analyze_candidates_header_info(self):
        # Mock a file
        test_filename = "test_case.txt"
        test_content = "This is the Header Information\nThis is the body content."
        with open(test_filename, "w", encoding="utf-8") as f:
            f.write(test_content)
        
        try:
            doc_id_map = {"test_case": os.path.abspath(test_filename)}
            
            # Mock _analyze_batch to just return the candidate with a dummy score
            # simulating that LLM selected it
            def mock_analyze_batch(query, cases, top_k=10):
                res = []
                for c in cases:
                    res.append({
                        'filename': c['filename'],
                        'similarity_score': 90,
                        'reason': 'Matched'
                    })
                return res
            
            self.rag._analyze_batch = mock_analyze_batch
            
            results = self.rag.analyze_candidates(doc_id_map, "query")
            
            self.assertEqual(len(results), 1)
            self.assertIn('header_info', results[0])
            self.assertEqual(results[0]['header_info'], "This is the Header Information")
            logger.info(f"SUCCESS: Header info extracted correctly: {results[0]['header_info']}")
            
        finally:
            if os.path.exists(test_filename):
                os.remove(test_filename)

if __name__ == '__main__':
    unittest.main()
