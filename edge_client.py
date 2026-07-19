import torch
import requests
import uuid
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
import time


MODEL = "Qwen/Qwen2.5-0.5B"
CLOUD_URL = "https://liquid-cycling-kindle.ngrok-free.dev" 

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
model.eval()

layers = model.model.layers
K = 4
session = requests.Session()

def edge_forward(input_ids, cache, past_len=0):
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
      position_embeddings=pos_emb
    )[0]
    
  return hidden, pos_ids

def generate(prompt, max_new_tokens=10):
    session_id = str(uuid.uuid4())
    session.post(f"{CLOUD_URL}/session_start", params={"session_id": session_id})

    input_ids = tok(prompt, return_tensors="pt").input_ids
    edge_cache = DynamicCache()
    cur_ids = input_ids
    past_len = 0
    generated = []
    
    step_times = []
    t_start = time.perf_counter()

    with torch.no_grad():
        for step in range(max_new_tokens):
            t0 = time.perf_counter()
            step_input = cur_ids if step == 0 else cur_ids[:, -1:]
            h, pos_ids = edge_forward(step_input, edge_cache, past_len)

            resp = session.post(f"{CLOUD_URL}/decode", json={
                "session_id": session_id,
                "hidden_state": h.flatten().tolist(),
                "shape": list(h.shape),
                "position_ids": pos_ids.flatten().tolist(),
            })
            next_token_id = resp.json()["next_token"]
            step_times.append(time.perf_counter() - t0)

            generated.append(next_token_id)
            next_token_tensor = torch.tensor([[next_token_id]])
            cur_ids = torch.cat([cur_ids, next_token_tensor], dim=1)
            past_len += step_input.shape[1]
    
    total = time.perf_counter() - t_start
    print(f"total: {total:.3f}s, per-token: {[f'{t:.3f}' for t in step_times]}")
    return tok.decode(input_ids[0].tolist() + generated)


if __name__ == "__main__":
    result = generate("The capital of France is", max_new_tokens=10)
    print("Result:", result)
