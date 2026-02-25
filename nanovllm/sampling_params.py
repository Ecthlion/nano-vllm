from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    task_str_len: int = 0
    task_type: str | None = None
    precision_tier: str = "balanced"
    seq_len_p95: int = 2048
    use_cmaes: bool = False

    def __post_init__(self):
        # assert self.temperature > 1e-10, "greedy sampling is not permitted"
        pass
