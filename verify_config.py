from modules.config_loader import load_config
import sys

try:
    config = load_config()
    print(f"Loaded Keys: {list(config.keys())}")
    
    # Check secrets
    openai_key = config.get('openai', {}).get('api_key', '')
    if len(openai_key) > 5:
        print(f"OpenAI Key loaded: {openai_key[:5]}...")
    else:
        print("OpenAI Key NOT loaded or empty")
        
    # Check base config
    data_dir = config.get('data_dir')
    print(f"Data Dir loaded: {data_dir}")
    
    if openai_key and data_dir:
        print("VERIFICATION SUCCESS")
        
    # Check embedding config
    emb_config = config.get('embedding', {})
    print(f"Embedding Config: {emb_config}")
    if emb_config.get('api_key'):
        print(f"Embedding API Key: {emb_config.get('api_key')[:5]}...")
        
    else:
        print("VERIFICATION FAILED")
        sys.exit(1)
        
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
