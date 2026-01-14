"""
案件类型识别模块
使用 LLM 自动识别用户查询对应的案件类型
"""
from openai import OpenAI
from .case_loader import load_config, get_available_types


def get_openai_client():
    """获取 OpenAI 客户端"""
    config = load_config()
    return OpenAI(
        api_key=config['openai']['api_key'],
        base_url=config['openai']['base_url']
    )


def classify_query(query: str) -> dict:
    """
    使用 LLM 识别用户查询对应的案件类型
    
    Args:
        query: 用户输入的查询文本
    
    Returns:
        {
            'case_type': 识别出的案件类型,
            'confidence': 置信度说明,
            'reason': 识别理由
        }
    """
    config = load_config()
    client = get_openai_client()
    
    available_types = get_available_types()
    if not available_types:
        # 如果没有实际数据，使用配置中的类型
        available_types = config.get('case_types', [])
    
    types_str = "、".join(available_types)
    
    prompt = f"""你是一个证券行政处罚案件分类专家。请根据用户的查询内容，判断其想要检索的案件类型。

可选的案件类型有：{types_str}

用户查询：{query}

请分析用户查询的关键词，判断最可能的案件类型。只输出 JSON 格式，包含以下字段：
- case_type: 案件类型（必须是上述可选类型之一）
- confidence: 置信度（高/中/低）
- reason: 判断理由（简短说明）

注意：
- 如果涉及内幕信息、提前买入卖出股票，属于"内幕交易"
- 如果涉及虚假陈述、财务造假、未及时披露，属于"信息披露违法违规"
- 如果涉及操纵股价、连续交易、对倒，属于"操纵市场"
- 如果涉及老鼠仓、利用职务便利获取信息，属于"利用未公开信息交易"

只输出 JSON，不要有其他内容。"""

    try:
        response = client.chat.completions.create(
            model=config['openai']['model'],
            messages=[
                {"role": "system", "content": "你是一个JSON输出助手，只输出有效的JSON格式。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        import json
        result = json.loads(response.choices[0].message.content)
        
        # 验证返回的类型是否有效
        if result.get('case_type') not in available_types and available_types:
            # 如果识别的类型不在可用列表中，选择第一个可用类型
            result['case_type'] = available_types[0]
            result['confidence'] = '低'
            result['reason'] = f"无法准确识别，默认使用 {available_types[0]}"
        
        return result
    
    except Exception as e:
        # 出错时返回默认值
        default_type = available_types[0] if available_types else "内幕交易"
        return {
            'case_type': default_type,
            'confidence': '低',
            'reason': f"识别出错: {str(e)}"
        }
