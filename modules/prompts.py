"""
提示词统一管理模块
此处集中管理程序中使用的所有提示词，方便直接编辑和修改。
提示词内容现在存储在 prompts/ 目录下的文本文件中。
"""
import os
from pathlib import Path
from loguru import logger

# 获取 prompts 目录的绝对路径
# 假设 current file is in modules/, so parent is project root
PROJECT_ROOT = Path(__file__).parent.parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"

def _load_prompt(filename: str) -> str:
    """从 prompts 目录加载提示词文件内容"""
    try:
        file_path = PROMPTS_DIR / filename
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error loading prompt '{filename}': {e}")
        return ""

# ==============================================================================
# 1. 查询拆解提示词 (Query Expansion)
# 文件: prompts/query_expansion.txt
# ==============================================================================
QUERY_EXPANSION_PROMPT = _load_prompt("query_expansion.txt")

# ==============================================================================
# 2. 相似案例检索提示词 (Similarity Search)
# 文件: prompts/similarity_search.txt
# ==============================================================================
SIMILARITY_SEARCH_PROMPT = _load_prompt("similarity_search.txt")

# ==============================================================================
# 3. 案件类型分类提示词 (Case Classification)
# 文件: prompts/case_classification.txt
# ==============================================================================
CASE_CLASSIFICATION_PROMPT = _load_prompt("case_classification.txt")
