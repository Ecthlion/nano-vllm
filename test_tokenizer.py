import pandas as pd
from transformers import AutoTokenizer, AutoConfig
import os

def calculate_kv_cache_size():
    model_path = "/data/zwt/model/models/Qwen/Qwen3-8B/"
    csv_path = "uploads/imdb.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print(f"Loading tokenizer and config from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        config = AutoConfig.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    df_filter = f"review.str.len() > 2500"
    df = df.query(df_filter)
    
    if 'review' not in df.columns:
        print("Error: 'review' column not found in CSV.")
        return
    
    print(f"Found {len(df)} reviews. Tokenizing...")
    
    # Batch tokenize for speed
    reviews = df['review'].astype(str).tolist()
    # We use the tokenizer to encode all texts. 
    # add_special_tokens=False because we just want to measure the content size roughly.
    encodings = tokenizer(reviews, add_special_tokens=False)
    total_tokens = sum(len(ids) for ids in encodings['input_ids'])
        
    print(f"Total tokens: {total_tokens}")
    
    # Calculate KV Cache per token
    # KV Cache Size (bytes) = 2 (K+V) * num_layers * num_kv_heads * head_dim * element_size
    # element_size = 2 bytes (float16/bfloat16)
    
    num_layers = config.num_hidden_layers
    # Use num_key_value_heads if available (GQA/MQA), else fallback to num_attention_heads
    num_kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    head_dim = hidden_size // num_attention_heads
    
    element_size = 2 # bytes for fp16/bf16
    
    kv_bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * element_size
    
    total_bytes = total_tokens * kv_bytes_per_token
    total_gb = total_bytes / (1024**3)
    
    print(f"Model Config:")
    print(f"  Layers: {num_layers}")
    print(f"  KV Heads: {num_kv_heads}")
    print(f"  Head Dim: {head_dim}")
    print(f"  Bytes per token (KV): {kv_bytes_per_token}")
    print("-" * 30)
    print(f"Total KV Cache Size: {total_gb:.4f} GB")

if __name__ == "__main__":
    calculate_kv_cache_size()
