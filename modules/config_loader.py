import yaml
import os
from pathlib import Path

def load_config():
    """
    加载配置文件
    1. 加载 config.yaml (基础配置)
    2. 加载 secrets.yaml (敏感配置，如果存在)
    3. 应用环境变量覆盖
    """
    base_path = Path(__file__).parent.parent
    
    # 1. 加载基础配置
    config_path = base_path / "config.yaml"
    config = {}
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}

    # 2. 加载 secrets.yaml
    secrets_path = base_path / "secrets.yaml"
    if secrets_path.exists():
        try:
            with open(secrets_path, 'r', encoding='utf-8') as f:
                secrets = yaml.safe_load(f) or {}
                # 简单合并 secrets 到 config (顶层键覆盖)
                # 注意：这里假设 secrets 中的结构与 config 互补或覆盖
                # 实际上 secrets 主要是 openai, analysis, embedding
                for key, value in secrets.items():
                    # 如果是字典且已存在，可以考虑深度合并，但这里 key 通常是顶层的大类 (openai 等)，直接覆盖或更新即可
                    if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                        config[key].update(value)
                    else:
                        config[key] = value
        except Exception as e:
            print(f"Warning: Failed to load secrets.yaml: {e}")

    # 3. 环境变量覆盖 (Render 部署使用)
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
    if 'embedding' not in config:
        config['embedding'] = {}
        
    # Get defaults from OpenAI config, but only use them if specific config is missing
    default_openai_key = config.get('openai', {}).get('api_key')
    default_openai_base = config.get('openai', {}).get('base_url')

    # API Key: Env -> Config -> Default
    env_emb_key = os.environ.get('EMBEDDING_API_KEY')
    if env_emb_key:
        config['embedding']['api_key'] = env_emb_key
    elif 'api_key' not in config['embedding']:
        config['embedding']['api_key'] = default_openai_key

    # Base URL: Env -> Config -> Default
    env_emb_base = os.environ.get('EMBEDDING_BASE_URL')
    if env_emb_base:
        config['embedding']['base_url'] = env_emb_base
    elif 'base_url' not in config['embedding']:
        config['embedding']['base_url'] = default_openai_base
        
    # Model: Env -> Config -> Default
    env_emb_model = os.environ.get('EMBEDDING_MODEL')
    if env_emb_model:
        config['embedding']['model'] = env_emb_model
    elif 'model' not in config['embedding']:
        config['embedding']['model'] = "text-embedding-v4"

    # Analysis Env Support
    if 'analysis' not in config:
        config['analysis'] = {}

    # API Key: Env -> Config -> Default
    env_ana_key = os.environ.get('ANALYSIS_API_KEY')
    if env_ana_key:
        config['analysis']['api_key'] = env_ana_key
    elif 'api_key' not in config['analysis']:
        config['analysis']['api_key'] = default_openai_key

    # Base URL: Env -> Config -> Default
    env_ana_base = os.environ.get('ANALYSIS_BASE_URL')
    if env_ana_base:
        config['analysis']['base_url'] = env_ana_base
    elif 'base_url' not in config['analysis']:
        config['analysis']['base_url'] = default_openai_base

    # Model: Env -> Config -> Default
    env_ana_model = os.environ.get('ANALYSIS_MODEL')
    if env_ana_model:
        config['analysis']['model'] = env_ana_model
    elif 'model' not in config['analysis']:
        config['analysis']['model'] = "gemini-1.5-pro"

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
