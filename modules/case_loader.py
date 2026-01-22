"""
案例数据加载模块
"""
import os
import yaml
from pathlib import Path



def load_config():
    """加载配置文件"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    
    # 默认从文件加载
    config = {}
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}

    # 环境变量覆盖 (Render 部署使用)
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    if openai_api_key:
        if 'openai' not in config:
            config['openai'] = {}
        config['openai']['api_key'] = openai_api_key

    openai_base_url = os.environ.get('OPENAI_BASE_URL')
    if openai_base_url:
        if 'openai' not in config:
            config['openai'] = {}
        config['openai']['base_url'] = openai_base_url
        
    openai_model = os.environ.get('OPENAI_MODEL')
    if openai_model:
        if 'openai' not in config:
            config['openai'] = {}
        config['openai']['model'] = openai_model

    # Embedding Env Support
    embedding_api_key = os.environ.get('EMBEDDING_API_KEY') or openai_api_key
    embedding_base_url = os.environ.get('EMBEDDING_BASE_URL') or openai_base_url
    embedding_model = os.environ.get('EMBEDDING_MODEL') or "text-embedding-v4"

    # Analysis Env Support
    analysis_api_key = os.environ.get('ANALYSIS_API_KEY') or openai_api_key
    analysis_base_url = os.environ.get('ANALYSIS_BASE_URL') or openai_base_url
    analysis_model = os.environ.get('ANALYSIS_MODEL') or "gemini-1.5-pro"

    if 'analysis' not in config:
        config['analysis'] = {}

    if analysis_api_key:
        config['analysis']['api_key'] = analysis_api_key
    if analysis_base_url:
        config['analysis']['base_url'] = analysis_base_url
    
    # Prioritize config file model if exists, else use env/default
    if 'model' not in config['analysis']:
        config['analysis']['model'] = analysis_model

    if 'embedding' not in config:
        config['embedding'] = {}
    
    if embedding_api_key:
        config['embedding']['api_key'] = embedding_api_key
    if embedding_base_url:
        config['embedding']['base_url'] = embedding_base_url
    config['embedding']['model'] = embedding_model

    # 设置默认值 (防止 config.yaml 不存在时报错)
    defaults = {
        'data_dir': 'data',
        'search': {
            'max_chars_per_request': 200000,
            'top_k': 10,
            'min_score': 60
        },
        'prompts': {
            'similarity_search_file': 'prompts/similarity_search.txt'
        },
        'case_types': []  # 如果没有配置类型，将自动从 data 目录读取
    }
    
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
        elif isinstance(value, dict) and isinstance(config.get(key), dict):
            # 简单的嵌套字典补全
            for sub_key, sub_value in value.items():
                if sub_key not in config[key]:
                    config[key][sub_key] = sub_value
        
    return config


def get_available_types():
    """获取所有可用的案件类型（基于 data 目录下的文件夹）"""
    config = load_config()
    data_dir = Path(__file__).parent.parent / config['data_dir']
    
    if not data_dir.exists():
        return []
    
    types = []
    for item in data_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # 检查文件夹内是否有 txt 文件
            txt_files = list(item.glob("*.txt"))
            if txt_files:
                types.append(item.name)
    
    # Check for direct files in data_dir (support for flat directory structure)
    if list(data_dir.glob("*.txt")):
        types.append("")
    
    return sorted(types)


def load_cases_by_type(case_type: str) -> "list[dict]":
    """
    加载指定类型的所有案例
    
    Args:
        case_type: 案件类型（对应 data 目录下的文件夹名）
    
    Returns:
        案例列表，每个案例包含 filename, content, char_count
    """
    config = load_config()
    data_dir = Path(__file__).parent.parent / config['data_dir'] / case_type
    
    if not data_dir.exists():
        return []
    
    cases = []
    for txt_file in sorted(data_dir.glob("*.txt")):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    cases.append({
                        'filename': txt_file.stem,  # 不含扩展名的文件名
                        'filepath': str(txt_file),
                        'content': content,
                        'char_count': len(content)
                    })
        except Exception as e:
            print(f"读取文件失败 {txt_file}: {e}")
    
    return cases


def get_cases_summary(case_type: str) -> dict:
    """
    获取指定类型案例的统计摘要
    """
    cases = load_cases_by_type(case_type)
    total_chars = sum(c['char_count'] for c in cases)
    
    return {
        'case_type': case_type,
        'case_count': len(cases),
        'total_chars': total_chars,
        'cases': [{'filename': c['filename'], 'char_count': c['char_count']} for c in cases]
    }
