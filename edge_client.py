import time
import uuid

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from config import CONFIG
from spec_decoding import draft_and_prepare_verification, run_edge_layers


tokenizer = AutoTokenizer.from_pretrained(CONFIG.model_name)
model = AutoModelForCausalLM.from_pretrained(
    CONFIG.model_name,
    torch_dtype=CONFIG.torch_dtype,
)
model.eval()
session = requests.Session()


def _post(path, **kwargs):
    response = session.post(
        f"{CONFIG.cloud_url.rstrip('/')}{path}",
        timeout=CONFIG.request_timeout_seconds,
        **kwargs,
    )
    response.raise_for_status()
    return response.json()


def _generate_standard(input_ids, max_new_tokens):
    session_id = str(uuid.uuid4())
    _post(
        "/session_start",
        params={"session_id": session_id, "edge_layers": CONFIG.edge_layers},
    )
    edge_cache = DynamicCache()
    cur_ids = input_ids
    past_len = 0
    generated = []
    round_times = []

    for step in range(max_new_tokens):
        started = time.perf_counter()
        step_input = cur_ids if step == 0 else cur_ids[:, -1:]
        hidden, position_ids = run_edge_layers(
            model, step_input, edge_cache, CONFIG.edge_layers, past_len
        )
        result = _post("/decode", json={
            "session_id": session_id,
            "hidden_state": hidden.flatten().tolist(),
            "shape": list(hidden.shape),
            "position_ids": position_ids.flatten().tolist(),
        })
        next_token = result["next_token"]
        generated.append(next_token)
        cur_ids = torch.cat(
            [cur_ids, torch.tensor([[next_token]], device=cur_ids.device)], dim=1
        )
        past_len += step_input.shape[1]
        round_times.append(time.perf_counter() - started)
        if next_token == tokenizer.eos_token_id:
            break

    return generated, round_times


def _generate_speculative(input_ids, max_new_tokens):
    cur_ids = input_ids
    generated = []
    round_times = []
    proposed_total = 0
    accepted_total = 0
    draft_seconds = 0.0
    preparation_seconds = 0.0
    http_seconds = 0.0
    acceptance_by_round = []

    while len(generated) < max_new_tokens:
        started = time.perf_counter()
        remaining = max_new_tokens - len(generated)
        draft_count = min(CONFIG.num_draft_tokens, remaining)
        draft_ids, hidden, position_ids, edge_timings = draft_and_prepare_verification(
            model, cur_ids, draft_count, CONFIG.edge_layers
        )
        draft_seconds += edge_timings["draft_seconds"]
        preparation_seconds += edge_timings["preparation_seconds"]

        request_started = time.perf_counter()
        result = _post("/verify", json={
            "hidden_state": hidden.flatten().tolist(),
            "shape": list(hidden.shape),
            "position_ids": position_ids.flatten().tolist(),
            "context_length": cur_ids.shape[1],
            "draft_ids": draft_ids[0].tolist(),
            "edge_layers": CONFIG.edge_layers,
        })
        http_seconds += time.perf_counter() - request_started

        accepted_count = result["accepted_count"]
        proposed_total += draft_count
        accepted_total += accepted_count
        acceptance_by_round.append(f"{accepted_count}/{draft_count}")
        accepted = draft_ids[0, :accepted_count].tolist()
        new_tokens = accepted + [result["bonus_token"]]
        new_tokens = new_tokens[:remaining]
        generated.extend(new_tokens)
        new_tensor = torch.tensor([new_tokens], device=cur_ids.device)
        cur_ids = torch.cat([cur_ids, new_tensor], dim=1)
        round_times.append(time.perf_counter() - started)

        if tokenizer.eos_token_id in new_tokens:
            eos_index = generated.index(tokenizer.eos_token_id)
            generated = generated[:eos_index + 1]
            break

    metrics = {
        "proposed": proposed_total,
        "accepted": accepted_total,
        "draft_seconds": draft_seconds,
        "preparation_seconds": preparation_seconds,
        "http_seconds": http_seconds,
        "acceptance_by_round": acceptance_by_round,
    }
    return generated, round_times, metrics


def generate(prompt, max_new_tokens=None):
    max_new_tokens = max_new_tokens or CONFIG.max_new_tokens
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    started = time.perf_counter()

    with torch.no_grad():
        if CONFIG.speculative_decoding:
            generated, round_times, metrics = _generate_speculative(
                input_ids, max_new_tokens
            )
            mode = "speculative"
        else:
            generated, round_times = _generate_standard(input_ids, max_new_tokens)
            mode = "standard"

    total = time.perf_counter() - started
    formatted_times = [f"{value:.3f}" for value in round_times]
    print(f"mode: {mode}, total: {total:.3f}s, rounds: {formatted_times}")
    if CONFIG.speculative_decoding:
        proposed = metrics["proposed"]
        accepted = metrics["accepted"]
        acceptance_rate = accepted / proposed if proposed else 0.0
        tokens_per_request = len(generated) / len(round_times) if round_times else 0.0
        print(
            "speculative metrics: "
            f"requests={len(round_times)}, proposed={proposed}, accepted={accepted}, "
            f"acceptance={acceptance_rate:.1%}, tokens/request={tokens_per_request:.2f}"
        )
        print(f"accepted by round: {metrics['acceptance_by_round']}")
        print(
            "timing breakdown: "
            f"draft={metrics['draft_seconds']:.3f}s, "
            f"verification_prep={metrics['preparation_seconds']:.3f}s, "
            f"http_and_serialization={metrics['http_seconds']:.3f}s"
        )
    return tokenizer.decode(input_ids[0].tolist() + generated)


if __name__ == "__main__":
    print("Result:", generate("The capital of France is"))
