def load_api_keys(file_path, model):
    MODEL_KEY_MAP = {
    "ChatGPT-4o-mini":"API_KEY_OPENAI",
    "llama-3.3-70b-versatile":"API_KEY_GROQ",
    "gpt-oss:120b":"API_KEY_ANVIL",
    "llama4:latest":"API_KEY_ANVIL",
    "gemma:latest":"API_KEY_ANVIL",
    }  
    keys = {}
    with open(file_path, 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                keys[key] = value
    
    required_key = MODEL_KEY_MAP.get(model)
    if required_key and required_key not in keys:
        raise ValueError(f"Required API key '{required_key}' for model '{model}' is missing in {file_path}")
    return keys


# Copyright Sep 2025 Glen Rogers. 
# Subject to MIT license.