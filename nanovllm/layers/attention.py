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
            # Discover pruning indices on the first layer only, and only when enabled.
            if (
                self.layer_id == 35
                and context.pruning_enabled
                and context.block_tables is None
                and context.cu_seqlens_q is not None
                and context.cu_seqlens_k is not None
            ):
                # Vectorized pruning index discovery on packed sequences.
                group_size = self.num_heads // self.num_kv_heads
                # q_gqa: [N, num_kv_heads, head_dim]
                q_gqa = q.view(q.size(0), self.num_kv_heads, group_size, self.head_dim).mean(dim=2)

                cuq = context.cu_seqlens_q  # [B+1]
                cuk = context.cu_seqlens_k  # [B+1]
                B = cuq.numel() - 1
                last_q_idx = cuq[1:] - 1  # [B]
                # Build sequence id per token in packed layout
                lens_k = (cuk[1:] - cuk[:-1]).to(torch.long)  # [B]
                seq_ids = torch.repeat_interleave(torch.arange(B, device=k.device), lens_k)

                # Gather last q per token's sequence and compute similarity
                q_last = q_gqa.index_select(0, last_q_idx)  # [B, num_kv_heads, head_dim]
                q_last_per_token = q_last.index_select(0, seq_ids)  # [N, num_kv_heads, head_dim]
                # sim per head, then mean over kv heads -> [N]
                scores = (k * q_last_per_token).sum(dim=-1).mean(dim=-1)

                alpha = context.sparsity
                pruned_locals: list[torch.Tensor] = []
                num_pruned = 0
                # Per-sequence topk on filtered scores
                for i in range(B):
                    s = int(cuk[i].item()); e = int(cuk[i+1].item())
                    seqlen_i = e - s
                    if seqlen_i <= 1:
                        pruned_locals.append(torch.empty(0, dtype=torch.int64, device=k.device))
                        continue
                    seq_scores = scores[s:e]
                    k_prune = max(int(alpha * seqlen_i), 0)
                    k_prune = min(k_prune, seq_scores.numel())
                    if k_prune <= 0:
                        pruned_locals.append(torch.empty(0, dtype=torch.int64, device=k.device))
                        continue
                    _, idx = torch.topk(seq_scores, k=k_prune, largest=False, sorted=False)
                    num_pruned += len(idx)
                    # store LOCAL indices within the sequence
                    pruned_locals.append(idx)

                print(f"[pruned] {num_pruned} tokens")

                # Stash into global context for later stages (KV cache store/persist)
                context.pruned_local_indices = pruned_locals

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
