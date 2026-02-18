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
        layer_id,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.layer_id = layer_id

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            # Discover pruning indices for each layer when pruning is enabled.
            if (
                context.pruning_enabled
                and context.block_tables is None
                and context.cu_seqlens_q is not None
                and context.cu_seqlens_k is not None
            ):
                # Vectorized pruning index discovery on packed sequences.
                group_size = self.num_heads // self.num_kv_heads
                q_gqa = q.view(q.size(0), self.num_kv_heads, group_size, self.head_dim).mean(dim=2)

                cuq = context.cu_seqlens_q
                cuk = context.cu_seqlens_k
                B = cuq.numel() - 1
                last_q_idx = cuq[1:] - 1
                lens_k = (cuk[1:] - cuk[:-1]).to(torch.long)
                seq_ids = torch.repeat_interleave(torch.arange(B, device=k.device), lens_k)

                q_last = q_gqa.index_select(0, last_q_idx)
                q_last_per_token = q_last.index_select(0, seq_ids)
                # Per-head importance scores [N, num_kv_heads].
                per_head_scores = (k * q_last_per_token).sum(dim=-1) * self.scale

                pruned_locals: list[torch.Tensor] = []
                for i in range(B):
                    s = int(cuk[i].item())
                    e = int(cuk[i + 1].item())
                    seqlen_i = e - s
                    if seqlen_i <= 1:
                        pruned_locals.append(torch.empty(0, dtype=torch.int64, device=k.device))
                        continue

                    seq_scores = per_head_scores[s : e - 1]
                    layer_head_sparsity = torch.full(
                        (self.num_kv_heads,),
                        float(context.sparsity),
                        dtype=seq_scores.dtype,
                        device=seq_scores.device,
                    )
                    if context.adaptive_sparsities is not None and i < len(context.adaptive_sparsities):
                        seq_sparse = context.adaptive_sparsities[i]
                        if seq_sparse.dim() == 2 and self.layer_id < seq_sparse.size(0):
                            layer_head_sparsity = seq_sparse[self.layer_id].to(
                                dtype=seq_scores.dtype, device=seq_scores.device
                            )
                    layer_head_sparsity = layer_head_sparsity.clamp_(0.50, 0.97)

                    keep_weight = (1.0 - layer_head_sparsity).unsqueeze(0)
                    weighted_scores = (seq_scores * keep_weight).mean(dim=-1)
                    prune_ratio = float(layer_head_sparsity.mean().item())
                    k_prune = max(int(prune_ratio * seqlen_i), 0)
                    k_prune = min(k_prune, weighted_scores.numel() - 1)
                    if k_prune <= 0:
                        pruned_locals.append(torch.empty(0, dtype=torch.int64, device=k.device))
                        continue
                    _, idx = torch.topk(weighted_scores, k=k_prune, largest=False, sorted=False)
                    pruned_locals.append(idx)

                # Stash into global context for later stages (KV cache store/persist).
                if context.pruned_local_indices is None:
                    context.pruned_local_indices = []
                while len(context.pruned_local_indices) <= self.layer_id:
                    context.pruned_local_indices.append([])
                context.pruned_local_indices[self.layer_id] = pruned_locals

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
