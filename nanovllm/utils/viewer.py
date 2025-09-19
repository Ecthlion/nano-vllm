import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from nanovllm.engine.sequence import Sequence


class KVCacheViewer:
    def __init__(self, kv_cache, config) -> None:
        assert config.tensor_parallel_size == 1
        self.kv_cache = kv_cache
        self.config = config
        self.seqs = []

    def add_seq(self, seq: Sequence):
        self.seqs.append(seq)

    def get_seq_kv_cache(self, seq: Sequence):
        hf_config = self.config.hf_config
        num_kv_heads = hf_config.num_key_value_heads
        num_layers = hf_config.num_hidden_layers

        self.local_kv_cache = torch.zeros(
            2,
            hf_config.num_hidden_layers,
            seq.num_tokens,
            num_kv_heads,
            hf_config.head_dim,
        )

        token_offset = 0
        transpose_kv_cache = self.kv_cache.transpose(1, 2)
        for block_id in seq.block_table:
            # Tensor[num_layers, block_size, num_heads, head_dim]
            block_k = transpose_kv_cache[0][block_id]
            block_v = transpose_kv_cache[1][block_id]

            if block_id == seq.block_table[-1]:
                block_tokens = seq.last_block_num_tokens
            else:
                block_tokens = block_k.shape[1]

            for layer_idx in range(num_layers):
                self.local_kv_cache[
                    0, layer_idx, token_offset : token_offset + block_tokens
                ] = block_k[layer_idx, :block_tokens]
                self.local_kv_cache[
                    1, layer_idx, token_offset : token_offset + block_tokens
                ] = block_v[layer_idx, :block_tokens]

            token_offset += block_tokens

    def visualize_3d_kv_cache(self, layer_idx=0, head_idx=0, cache_type="value"):
        if cache_type == "key":
            data = (
                self.local_kv_cache[0, layer_idx, :, head_idx].cpu().numpy()
            )  # [tokens, head_dim]
        else:  # value
            data = (
                self.local_kv_cache[1, layer_idx, :, head_idx].cpu().numpy()
            )  # [tokens, head_dim]

        tokens, head_dim = data.shape

        x = np.arange(tokens)
        y = np.arange(head_dim)
        X, Y = np.meshgrid(y, x)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        surf = ax.plot_surface(X, Y, data, cmap="viridis", alpha=0.8, edgecolor="none")

        ax.set_xlabel("Hidden Dimension", fontsize=12, labelpad=10)
        ax.set_ylabel("Token Position", fontsize=12, labelpad=10)
        ax.set_zlabel("Activation Value", fontsize=12, labelpad=10)

        title = f"3D {cache_type.capitalize()} Cache - Layer {layer_idx}, Head {head_idx}\nSeq Length: {tokens}"
        ax.set_title(title, fontsize=14, pad=20)

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label="Activation Value")

        ax.view_init(elev=30, azim=45)

        plt.tight_layout()
        plt.savefig(f"./graph/3d_{cache_type}_layer{layer_idx}_head{head_idx}.png")
        plt.close()

        return fig, ax

    def visualize_heatmap_kv_cache(self, layer_idx=0, head_idx=0, cache_type="value"):
        if cache_type == "key":
            data = (
                self.local_kv_cache[0, layer_idx, :, head_idx].cpu().numpy()
            )  # [tokens, head_dim]
        else:  # value
            data = (
                self.local_kv_cache[1, layer_idx, :, head_idx].cpu().numpy()
            )  # [tokens, head_dim]

        fig, ax = plt.subplots(figsize=(12, 8))

        # 使用seaborn绘制热力图
        heatmap = sns.heatmap(
            data,
            cmap="RdBu_r",  # 红蓝渐变色，适合显示正负值
            center=0,  # 以0为中心
            ax=ax,
            cbar_kws={"label": "Activation Value"},
        )

        # 设置标签
        ax.set_xlabel("Hidden Dimension", fontsize=12)
        ax.set_ylabel("Token Position", fontsize=12)

        title = f"Heatmap of {cache_type.capitalize()} Cache - Layer {layer_idx}, Head {head_idx}\nSeq Length: {data.shape[0]}"
        ax.set_title(title, fontsize=14, pad=20)

        # 如果token数量很多，可以适当减少刻度显示
        if data.shape[0] > 50:
            ax.set_yticks(np.linspace(0, data.shape[0], 10, dtype=int))
        if data.shape[1] > 50:
            ax.set_xticks(np.linspace(0, data.shape[1], 10, dtype=int))

        plt.tight_layout()
        plt.savefig(f"./graph/heatmap_{cache_type}_layer{layer_idx}_head{head_idx}.png")
        plt.close()

        return fig, ax

    def view(self, seq_idx=0):
        if not self.seqs:
            print("No sequences added for visualization.")
            return

        seq = self.seqs[seq_idx]
        self.get_seq_kv_cache(seq)

        print(f"Visualizing KV cache for sequence with {seq.num_tokens} tokens")

        print("Generating 3D visualization...")
        for layer_idx in range(0, self.config.hf_config.num_hidden_layers, 5):
            self.visualize_3d_kv_cache(layer_idx, 0, "key")
            self.visualize_3d_kv_cache(layer_idx, 4, "key")
            self.visualize_3d_kv_cache(layer_idx, 0, "value")
            self.visualize_3d_kv_cache(layer_idx, 4, "value")

        print("Generating heatmap visualization...")
        for layer_idx in range(0, self.config.hf_config.num_hidden_layers, 5):
            self.visualize_heatmap_kv_cache(layer_idx, 0, "key")
            self.visualize_heatmap_kv_cache(layer_idx, 4, "key")
            self.visualize_heatmap_kv_cache(layer_idx, 0, "value")
            self.visualize_heatmap_kv_cache(layer_idx, 4, "value")

        # print("Generating multi-head overview...")
        # self.visualize_all_heads(layer_idx, "key")
