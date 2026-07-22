"""Cloud-side server for split and speculative LLM inference."""

import importlib.util
import inspect
import os
import subprocess
import sys
import threading


def _install_missing_dependencies():
    """Install Kaggle runtime dependencies before importing them."""
    packages = {
        "accelerate": "accelerate",
        "fastapi": "fastapi",
        "pyngrok": "pyngrok",
        "transformers": "transformers",
        "uvicorn": "uvicorn[standard]",
    }
    missing = [
        package
        for module, package in packages.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        print(f"Installing missing dependencies: {', '.join(missing)}")
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            *missing,
        ])


_install_missing_dependencies()

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pyngrok import ngrok
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DEVICE = os.getenv("DEVICE", "cuda")

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
).to(DEVICE)
model.eval()
layers = model.model.layers

app = FastAPI()

# session_id -> cache and model-split metadata
sessions = {}


def _get_ngrok_auth_token():
    """Read ngrok credentials from the environment or Kaggle Secrets."""
    auth_token = os.getenv("AUTH_TOKEN")
    if auth_token:
        return auth_token

    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret("AUTH_TOKEN")
    except Exception as error:
        raise RuntimeError(
            "Add an AUTH_TOKEN Kaggle secret containing your ngrok token, "
            "or set the AUTH_TOKEN environment variable"
        ) from error


class DecodeRequest(BaseModel):
    session_id: str
    hidden_state: list[float]
    shape: list[int]
    position_ids: list[int]


class VerifyRequest(BaseModel):
    hidden_state: list[float]
    shape: list[int]
    position_ids: list[int]
    context_length: int
    draft_ids: list[int]
    edge_layers: int


def _layer_hidden(layer_output):
    return layer_output[0] if isinstance(layer_output, tuple) else layer_output


def _cache_argument(layer, cache):
    """Support both old and new Transformers decoder-layer cache APIs."""
    parameters = inspect.signature(layer.forward).parameters
    if "past_key_values" in parameters:
        return {"past_key_values": cache}
    if "past_key_value" in parameters:
        return {"past_key_value": cache}
    raise RuntimeError("Decoder layer does not expose a supported KV-cache argument")


def _causal_attention_mask(hidden, position_ids, past_len):
    """Return a 4D additive causal mask for direct decoder-layer calls."""
    sequence_length = hidden.shape[1]
    key_length = past_len + sequence_length
    key_positions = torch.arange(key_length, device=hidden.device).view(1, 1, -1)
    blocked = key_positions > position_ids.unsqueeze(-1)
    mask = torch.zeros(
        (hidden.shape[0], 1, sequence_length, key_length),
        dtype=hidden.dtype,
        device=hidden.device,
    )
    return mask.masked_fill(blocked.unsqueeze(1), torch.finfo(hidden.dtype).min)


def _validate_edge_layers(edge_layers):
    if not 0 <= edge_layers <= len(layers):
        raise HTTPException(
            status_code=400,
            detail=f"edge_layers must be between 0 and {len(layers)}",
        )


def _request_tensors(hidden_state, shape, position_ids):
    try:
        hidden = torch.tensor(
            hidden_state,
            dtype=torch.float32,
            device=DEVICE,
        ).reshape(shape)
        positions = torch.tensor(
            position_ids,
            dtype=torch.long,
            device=DEVICE,
        ).reshape(1, -1)
    except (RuntimeError, TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    if hidden.ndim != 3 or positions.shape[1] != hidden.shape[1]:
        raise HTTPException(
            status_code=400,
            detail="Hidden states and position IDs have incompatible shapes",
        )
    return hidden, positions


def cloud_forward(hidden, position_ids, cache, edge_layers, past_len):
    position_embeddings = model.model.rotary_emb(hidden, position_ids)
    attention_mask = _causal_attention_mask(hidden, position_ids, past_len)
    for layer in layers[edge_layers:]:
        hidden = _layer_hidden(layer(
            hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            position_embeddings=position_embeddings,
            **_cache_argument(layer, cache),
        ))
    hidden = model.model.norm(hidden)
    return model.lm_head(hidden)


@app.post("/session_start")
async def session_start(session_id: str, edge_layers: int):
    _validate_edge_layers(edge_layers)
    sessions[session_id] = {
        "cache": DynamicCache(),
        "past_len": 0,
        "edge_layers": edge_layers,
    }
    return {"status": "session created"}


@app.post("/decode")
async def decode(request: DecodeRequest):
    session = sessions.get(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session_id")

    hidden, position_ids = _request_tensors(
        request.hidden_state,
        request.shape,
        request.position_ids,
    )
    with torch.no_grad():
        logits = cloud_forward(
            hidden,
            position_ids,
            session["cache"],
            session["edge_layers"],
            session["past_len"],
        )
        next_token = logits[:, -1, :].argmax(-1).item()

    session["past_len"] += hidden.shape[1]
    return {"next_token": next_token}


@app.post("/verify")
async def verify(request: VerifyRequest):
    _validate_edge_layers(request.edge_layers)
    if request.context_length < 1 or not request.draft_ids:
        raise HTTPException(
            status_code=400,
            detail="Verification requires a context and at least one draft token",
        )

    hidden, position_ids = _request_tensors(
        request.hidden_state,
        request.shape,
        request.position_ids,
    )
    expected_length = request.context_length + len(request.draft_ids)
    if hidden.shape[1] != expected_length:
        raise HTTPException(
            status_code=400,
            detail="Hidden-state length does not match context plus drafts",
        )

    with torch.no_grad():
        logits = cloud_forward(
            hidden,
            position_ids,
            DynamicCache(),
            request.edge_layers,
            0,
        )
        predicted = logits[0].argmax(-1)

    accepted_count = 0
    for index, draft_id in enumerate(request.draft_ids):
        verifier_id = predicted[request.context_length - 1 + index].item()
        if verifier_id != draft_id:
            break
        accepted_count += 1

    bonus_index = request.context_length - 1 + accepted_count
    return {
        "accepted_count": accepted_count,
        "bonus_token": predicted[bonus_index].item(),
    }


@app.get("/ping")
async def ping():
    return {
        "status": "alive",
        "model": MODEL_NAME,
        "device": DEVICE,
        "num_layers": len(layers),
    }


def main():
    ngrok.set_auth_token(_get_ngrok_auth_token())
    ngrok.kill()
    public_url = ngrok.connect(PORT).public_url
    print(f"Public URL: {public_url}")
    print(f"Ping URL: {public_url}/ping")

    config = uvicorn.Config(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
    )
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, name="uvicorn-server")
    try:
        # Kaggle/IPython already has an asyncio event loop. Running Uvicorn in
        # its own thread gives it a separate loop and also works as a .py file.
        server_thread.start()
        server_thread.join()
    except KeyboardInterrupt:
        server.should_exit = True
        server_thread.join()
    finally:
        ngrok.kill()


if __name__ == "__main__":
    main()
