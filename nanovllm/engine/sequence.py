from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(
        self,
        token_ids: list[int] | tuple[int, list[int]],
        text_token_len=0,
        pruning_len=0,
        sampling_params=SamplingParams(),
        task_type: str = "generic",
        precision_tier: str = "balanced",
        adaptive_sparsity: list[list[float]] | None = None,
    ):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        if isinstance(token_ids, tuple):
            self.text_id = token_ids[0]
            self.token_ids = copy(token_ids[1])
        else:
            self.text_id = None
            self.token_ids = copy(token_ids)

        self.last_token = self.token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids[1]) if isinstance(token_ids, tuple) else len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.lock_block = False
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.text_token_len = text_token_len
        # Pruning related fields: list of per-layer local index lists
        self.pruning_indices: list[list[int]] = []
        self.pruning_len = pruning_len
        self.task_type = task_type
        self.precision_tier = precision_tier
        self.adaptive_sparsity = adaptive_sparsity

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
