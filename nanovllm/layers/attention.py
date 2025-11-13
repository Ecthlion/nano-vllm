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

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, token_types: torch.Tensor | None = None):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
            
        if context.is_prefill:
            if token_types is not None and context.cu_seqlens_q is not None:
                o = self.optimized_attention(q, k, v, token_types, context)
            else:
                if context.block_tables is not None:    # prefix cache
                    k, v = k_cache, v_cache
                o = flash_attn_varlen_func(q, k, v,
                                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o

    def optimized_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                            token_types: torch.Tensor, context):
        
        data_mask = token_types == 0
        task_mask = token_types == 1
        
        output = torch.zeros_like(q)
        if data_mask.any():
            if v.shape[1] != q.shape[1]:
                repeat_factor = q.shape[1] // v.shape[1]
                v_expanded = v.repeat_interleave(repeat_factor, dim=1)
            else:
                v_expanded = v
            output[data_mask] = v_expanded[data_mask]
        
        if task_mask.any():
            task_output = self.compute_task_attention(
                q[task_mask], k, v
            )
            output[task_mask] = task_output
        
        return output

    def compute_task_attention(self, task_q: torch.Tensor, all_k: torch.Tensor, all_v: torch.Tensor):   
        num_tasks = task_q.shape[0]
        total_tokens = all_k.shape[0]
        
        cu_seqlens_q = torch.tensor([0, num_tasks], device=task_q.device, dtype=torch.int32)
        cu_seqlens_k = torch.tensor([0, total_tokens], device=all_k.device, dtype=torch.int32)
        
        result = flash_attn_varlen_func(
            task_q, 
            all_k, 
            all_v, 
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=num_tasks,
            max_seqlen_k=total_tokens,
            softmax_scale=self.scale,
            causal=False
        )
        return result


