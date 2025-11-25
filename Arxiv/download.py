import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import load_dataset
custom_cache_dir = "/data/zwt/huggingface/datasets"
dataset = load_dataset("leminda-ai/s2orc_small", cache_dir=custom_cache_dir)

