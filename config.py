"""Configuration shared by the edge inference entry point."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class InferenceConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B"
    cloud_url: str = "https://liquid-cycling-kindle.ngrok-free.dev"
    edge_layers: int = 12
    speculative_decoding: bool = False
    num_draft_tokens: int = 3
    max_new_tokens: int = 10
    request_timeout_seconds: float = 60.0
    torch_dtype: torch.dtype = torch.float32


CONFIG = InferenceConfig()
