"""
结果归并排序模块
"""


def merge_and_sort(results: list[dict], min_score: int = 60) -> list[dict]:
    """
    归并多批结果，按相似度降序排列，只保留分数>=min_score的结果
    
    Args:
        results: 所有批次的检索结果
        min_score: 最低分数阈值（默认60分）
    
    Returns:
        排序后的案例列表
    """
    if not results:
        return []
    
    # 按相似度降序排序
    sorted_results = sorted(
        results, 
        key=lambda x: x.get('similarity_score', 0), 
        reverse=True
    )
    
    # 去重（同一个案例可能在多批次中出现的情况较少，但以防万一）
    seen_filenames = set()
    unique_results = []
    
    for r in sorted_results:
        filename = r.get('filename', '')
        score = r.get('similarity_score', 0)
        # 只保留分数>=min_score的结果
        if filename and filename not in seen_filenames and score >= min_score:
            seen_filenames.add(filename)
            unique_results.append(r)
    
    return unique_results


def format_results_for_display(results: list[dict]) -> list[dict]:
    """
    格式化结果用于前端展示
    """
    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append({
            'rank': i,
            'filename': r.get('filename', '未知'),
            'similarity_score': r.get('similarity_score', 0),
            'summary': r.get('summary', ''),
            'reason': r.get('reason', ''),
            'content': r.get('content', ''),
            'filepath': r.get('filepath', '')
        })
    return formatted
