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
    
    return sorted(types)


def load_cases_by_type(case_type: str) -> list[dict]:
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
