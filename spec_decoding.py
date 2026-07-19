import torch
import requests
import uuid
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
import time

def draft_early_exit(model, cur_ids, num_draft):
    # run embed_tokens + layers[:K] + norm + lm_head, greedily,
    # num_draft times, collecting predicted token ids
    # (start with NO cache here — just recompute each time, simplicity first)
    K = 5
    past_len = 0
    cache = DynamicCache()
    
    draft_ids = None
    for i in range(num_draft):
      input = cur_ids if i == 0 else draft_ids[:, -1:]
      logits = forward(model, input, cache, K, past_len)
      next_token = logits[:, -1, :].argmax(-1, keepdim=True)
      draft_ids = next_token if draft_ids is None else torch.cat([draft_ids, next_token], dim=1)
      past_len += input.shape[1]


    return draft_ids  # tensor of shape (1, num_draft)

def verify_full_model(model, cur_ids, draft_ids):
    total_ids = torch.cat([cur_ids, draft_ids], dim=1)
    cache = DynamicCache()
    layers = model.model.layers
    logits = forward(model, total_ids, cache, len(layers), 0)

    N = cur_ids.shape[1]
    num_draft = draft_ids.shape[1]
    predicted = logits[0].argmax(-1)     # shape (N + num_draft,)
    draft_list = draft_ids[0].tolist()

    accepted = 0
    for i in range(num_draft):
        if predicted[N - 1 + i].item() != draft_list[i]:
            break
        accepted += 1

    accepted_ids = draft_ids[:, :accepted]
    bonus_token = predicted[N - 1 + accepted].view(1, 1)

    return accepted_ids, bonus_token


def forward(model, input_ids, cache, K, past_len=0):
    layers = model.model.layers
    hidden = model.model.embed_tokens(input_ids)
    seq_len = hidden.shape[1]
    pos_ids = torch.arange(past_len, past_len + seq_len).unsqueeze(0)
    pos_emb = model.model.rotary_emb(hidden, pos_ids)
    
    for layer in layers[:K]:
        hidden = layer(
            hidden,
            position_ids=pos_ids,
            past_key_value=cache,
            use_cache=True,
            position_embeddings=pos_emb,
        )[0]
    hidden = model.model.norm(hidden)
    logits = model.lm_head(hidden)
    return logits