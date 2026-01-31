
import sys
import os
from loguru import logger
sys.path.append(os.getcwd())
from modules.config_loader import load_config
import pprint

def verify():
    config = load_config()
    logger.info("Loaded Config:")
    logger.info(f"\n{pprint.pformat(config)}")
    
    # Check OpenAI
    assert config['openai']['base_url'] == "https://api.qiyiguo.uk/v1"
    assert config['openai']['query_model'] == "[vt-按量计费]gemini-3-pro-preview"
    assert config['openai']['temperature'] == 0
    assert 'api_key' in config['openai']
    
    # Check Analysis
    assert config['analysis']['base_url'] == "https://api.qiyiguo.uk/v1"
    assert config['analysis']['model'] == "[1000k按次计费]gemini-3-pro-preview"
    assert 'api_key' in config['analysis']
    
    # Check Embedding
    assert config['embedding']['base_url'] == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert config['embedding']['model'] == "text-embedding-v4"
    assert 'api_key' in config['embedding']
    
    logger.info("\nVerification Successful!")

if __name__ == "__main__":
    verify()
