"""Configuration shared by the edge inference entry point."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class InferenceConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B"
    cloud_url: str = "https://liquid-cycling-kindle.ngrok-free.dev"
    split_inference: bool = True
    edge_layers: int = 20
    speculative_decoding: bool = True
    num_draft_tokens: int = 3
    max_new_tokens: int = 10
    request_timeout_seconds: float = 60.0
    torch_dtype: torch.dtype = torch.float32
    activation_dtype: str = "fp16"  # "fp32", "fp16", or "int4"

    @property
    def cloud_ws_url(self):
        return self.cloud_url.rstrip("/").replace("https://", "wss://").replace("http://", "ws://") + "/session"


CONFIG = InferenceConfig()
