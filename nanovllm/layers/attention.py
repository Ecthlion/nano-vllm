import time
import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context

@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)

def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)

class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.head_to_kv = torch.arange(num_heads) % num_kv_heads

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, token_types: torch.Tensor | None = None, previous_q_ready=None):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
            
        if context.is_prefill:
            # 在prefill阶段使用优化的注意力计算
            if token_types is not None and context.cu_seqlens_q is not None and previous_q_ready is not None:
                
                # 跳过data token
                # torch.cuda.synchronize()
                # start_time = time.perf_counter()
                task_mask = token_types == 1
                task_q = q[task_mask]        
                if task_mask.any():
                    task_output = flash_attn_varlen_func(
                        task_q, 
                        k, 
                        v, 
                        cu_seqlens_q=context.cu_seqlens_taskq,
                        cu_seqlens_k=context.cu_seqlens_k,
                        max_seqlen_q=task_q.shape[0],
                        max_seqlen_k=k.shape[0],
                        softmax_scale=self.scale,
                        causal=True
                    )
                    previous_q_ready[task_mask] = task_output
                o = previous_q_ready
                # torch.cuda.synchronize()
                # end_time = time.perf_counter()
                # print(f"优化分支时间: {(end_time - start_time)*1000:.2f}ms")
            else:
                # 回退到原始实现
                if context.block_tables is not None:    # prefix cache
                    k, v = k_cache, v_cache
                # torch.cuda.synchronize()
                # start_time = time.perf_counter()
                o = flash_attn_varlen_func(q, k, v,
                                        max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                        max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                        softmax_scale=self.scale, causal=True, block_table=context.block_tables)
                # torch.cuda.synchronize()
                # end_time = time.perf_counter()
                # print(f"标准分支时间: {(end_time - start_time)*1000:.2f}ms")
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o