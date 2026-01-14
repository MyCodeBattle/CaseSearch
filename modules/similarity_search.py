"""
LLM 相似度检索模块
通过全量对比方式检索相似案例
"""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from .case_loader import load_config


def get_openai_client():
    """获取 OpenAI 客户端"""
    config = load_config()
    return OpenAI(
        api_key=config['openai']['api_key'],
        base_url=config['openai']['base_url']
    )


def batch_cases_by_chars(cases: list[dict], max_chars: int) -> list[list[dict]]:
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


def search_similar_in_batch(query: str, cases: list[dict], top_k: int = 10) -> list[dict]:
    """
    在一批案例中检索最相似的案例
    
    Args:
        query: 用户查询
        cases: 案例列表
        top_k: 返回的最相似案例数量
    
    Returns:
        相似案例列表，包含 filename, similarity_score, summary, reason
    """
    config = load_config()
    client = get_openai_client()
    
    # 构建案例文本
    cases_text = ""
    for i, case in enumerate(cases, 1):
        cases_text += f"\n\n===== 案例 {i}: {case['filename']} =====\n{case['content']}\n"
    
    # 加载提示词：优先从外部文件读取，否则使用默认模板
    default_prompt = """你是一个证券行政处罚案例检索专家。请根据用户的查询，从以下案例中找出最相似的案例。

用户查询：{query}

以下是待检索的案例：
{cases_text}

请分析每个案例与用户查询的相似度，考虑以下因素：
1. 违法行为类型是否相同
2. 违法情节是否相似（如获利金额、持续时间、手段方法等）
3. 涉及主体是否类似（个人/机构）
4. 处罚结果是否相近

请返回最相似的 {top_k} 个案例，按相似度从高到低排序。

输出 JSON 格式，包含一个 results 数组，每个元素包含：
- filename: 案例文件名（必须与上面提供的案例名称完全一致）
- similarity_score: 相似度评分（0-100的整数，100表示完全相似）
- summary: 案例摘要（50字以内概括该案例的核心内容）
- reason: 相似理由（说明为什么这个案例与查询相似）

只输出 JSON，不要有其他内容。"""
    
    # 尝试从外部文件读取提示词
    prompt_template = default_prompt
    prompts_config = config.get('prompts', {})
    
    # 支持文件路径方式（新方式）
    prompt_file = prompts_config.get('similarity_search_file')
    if prompt_file:
        import os
        # 获取项目根目录（config.yaml 所在目录）
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        prompt_path = os.path.join(project_root, prompt_file)
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt_template = f.read()
        except Exception as e:
            print(f"读取提示词文件失败，使用默认提示词: {e}")
    # 兼容旧的内联方式
    elif prompts_config.get('similarity_search'):
        prompt_template = prompts_config.get('similarity_search')
    
    # 填充模板变量
    prompt = prompt_template.format(
        query=query,
        cases_text=cases_text,
        top_k=min(top_k, len(cases))
    )

    try:
        response = client.chat.completions.create(
            model=config['openai']['model'],
            messages=[
                {"role": "system", "content": "你是一个JSON输出助手，只输出有效的JSON格式。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        # 获取响应内容
        content = response.choices[0].message.content
        if not content:
            print("警告: API 返回了空内容")
            return []
        
        # 清理可能存在的 Markdown 代码块标记
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]  # 移除 ```json
        elif content.startswith("```"):
            content = content[3:]  # 移除 ```
        if content.endswith("```"):
            content = content[:-3]  # 移除结尾的 ```
        content = content.strip()
        
        # 尝试解析 JSON
        try:
            result = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"JSON 解析失败: {e}")
            print(f"原始响应内容: {content[:500]}")  # 只打印前500字符
            return []
        
        results = result.get('results', [])
        
        # 验证并补充案例信息
        valid_filenames = {c['filename']: c for c in cases}
        valid_results = []
        
        for r in results:
            filename = r.get('filename', '')
            if filename in valid_filenames:
                r['filepath'] = valid_filenames[filename]['filepath']
                r['content'] = valid_filenames[filename]['content']
                valid_results.append(r)
        
        return valid_results[:top_k]
    
    except Exception as e:
        print(f"检索出错: {e}")
        return []


def search_similar_cases(query: str, cases: list[dict], progress_callback=None) -> list[dict]:
    """
    检索相似案例（支持并发分批处理）
    
    Args:
        query: 用户查询
        cases: 所有案例列表
        progress_callback: 进度回调函数，接收 (completed, total, batch_results) 参数
    
    Returns:
        相似案例列表（已按相似度排序）
    """
    config = load_config()
    max_chars = config['search']['max_chars_per_request']
    top_k = config['search']['top_k']
    
    # 按字数分批
    batches = batch_cases_by_chars(cases, max_chars)
    total_batches = len(batches)
    
    print(f"共 {len(cases)} 个案例，分为 {total_batches} 批并发处理")
    
    all_results = []
    completed_count = 0
    
    # 串行处理（适应 Render 免费版单线程限制）
    for i, batch in enumerate(batches, 1):
        try:
            batch_results = search_similar_in_batch(query, batch, top_k)
            all_results.extend(batch_results)
            completed_count += 1
            print(f"批次 {i}/{total_batches} 完成，获得 {len(batch_results)} 个结果")
            
            # 调用进度回调
            if progress_callback:
                progress_callback(completed_count, total_batches, batch_results)
        except Exception as e:
            completed_count += 1
            print(f"批次 {i} 处理出错: {e}")
            if progress_callback:
                progress_callback(completed_count, total_batches, [])
    
    return all_results
