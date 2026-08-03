# llama.cpp on RunPod Serverless

> Run any GGUF model as an OpenAI-compatible API on RunPod Serverless GPUs — zero infrastructure to manage.

This project packages [llama.cpp](https://github.com/ggml-org/llama.cpp) as a [RunPod Serverless](https://www.runpod.io/serverless-gpu) worker. A lightweight Python handler boots `llama-server` inside a CUDA-enabled container, proxies inference requests through an OpenAI-compatible interface, and scales to zero when idle.

## Features

- **OpenAI-compatible API** — Drop-in replacement for `/v1/chat/completions`, `/v1/completions`, and `/v1/models`
- **Any GGUF model** — Load models via direct URL, Hugging Face Hub, or a local/network-volume path
- **Vision support** — Optional mmproj projector loading via `MMPROJ_URL` or `MMPROJ_PATH`
- **Thinking mode control** — Disable/enable hybrid reasoning models (e.g. Qwen3.6) via `LLAMA_ENABLE_THINKING`
- **Full GPU offload** — Runs on the `llama.cpp` CUDA backend with configurable GPU layers
- **Auto-lifecycle management** — `llama-server` starts on first request, shuts down on container exit
- **Scale to zero** — RunPod Serverless billing only when actively processing requests
- **Flexible payload** — Send a simple `prompt` string or a full OpenAI-style `payload` object

## Architecture

![Architecture Diagram](./docs/diagrams/architecture.svg)

The worker runs inside a RunPod Serverless container. When a job arrives:

1. **RunPod Gateway** dispatches the job to an available worker container.
2. **`handler.py`** ensures `llama-server` is running (starts it on first call).
3. The handler builds an OpenAI-compatible payload and POSTs it to `localhost:8080`.
4. **`llama-server`** runs inference on the GPU and returns the result.
5. The handler wraps the response and returns it through RunPod to the client.

## Data Flow

![Data Flow Diagram](./docs/diagrams/data-flow.svg)

## Quick Start

### Prerequisites

- Docker with NVIDIA GPU support ([nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))
- A [RunPod](https://www.runpod.io/) account (for deployment)
- A GGUF model (the default pulls one from Hugging Face automatically via `MODEL_URL`)

### Build the Image

```bash
docker build -t <your-dockerhub-user>/llama-cpp-runpod:latest .
docker push <your-dockerhub-user>/llama-cpp-runpod:latest
```

### Deploy to RunPod

1. Create a **Serverless Template** in the RunPod console with your image.
2. Choose a GPU with enough VRAM for your model (e.g., RTX 5090 24 GB+ for 27B Q5).
3. Set environment variables (see [Configuration](#configuration)).
4. Create an **Endpoint** from the template.
5. Send requests to `https://api.runpod.ai/v2/<endpoint-id>/runsync`.

See [`runpod-endpoint-config.example.json`](runpod-endpoint-config.example.json) for a sample endpoint configuration.

## Request Format

Requests are sent as RunPod jobs via `/run` (async) or `/runsync` (synchronous).

### Simple prompt

```json
{
  "input": {
    "prompt": "Explain why GGUF is used with llama.cpp in 3 bullet points.",
    "temperature": 0.2,
    "max_tokens": 200
  }
}
```

### Full OpenAI-compatible payload

```json
{
  "input": {
    "endpoint": "/v1/chat/completions",
    "payload": {
      "model": "gpt-4o",
      "messages": [
        {"role": "user", "content": "Say hello in one sentence."}
      ],
      "temperature": 0.7,
      "max_tokens": 64,
      "stream": false
    }
  }
}
```

### Response structure

```json
{
  "status_code": 200,
  "ok": true,
  "endpoint": "/v1/chat/completions",
  "result": { "...": "OpenAI-compatible response from llama-server" }
}
```

### Supported optional parameters

When using the simple format (without `payload`), these keys are forwarded:

| Parameter | Description |
|-----------|-------------|
| `messages` | Full chat messages array (overrides `prompt`) |
| `model` | Model alias to use |
| `temperature` | Sampling temperature |
| `top_p` | Nucleus sampling threshold |
| `top_k` | Top-k sampling |
| `max_tokens` | Maximum tokens to generate |
| `presence_penalty` | Presence penalty |
| `frequency_penalty` | Frequency penalty |
| `stop` | Stop sequences |
| `response_format` | Response format (e.g., JSON mode) |
| `tools` | Function/tool definitions |
| `tool_choice` | Tool selection strategy |
| `params` | Dict of additional parameters merged into payload |

## Configuration

All configuration is via environment variables, set either in the Dockerfile or in the RunPod endpoint settings. RunPod endpoint env vars take priority over Dockerfile defaults.

### Model configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Local path to a GGUF file (highest priority) | — |
| `MODEL_URL` | Direct URL to a GGUF file (e.g. HF resolve URL) | *(set in Dockerfile)* |
| `MODEL_HF_REPO` | Hugging Face repo containing GGUF files (fallback) | — |
| `MODEL_HF_FILE` | Specific file within the HF repo | — |
| `MODEL_ALIAS` | Model name returned in API responses | `gpt-4o` |
| `MMPROJ_URL` | Direct URL to an mmproj file for vision support | *(set in Dockerfile)* |
| `MMPROJ_PATH` | Local path to an mmproj file | — |
| `HF_TOKEN` | Hugging Face token for gated/private repos | — |

> **Priority order:** `MODEL_PATH` → `MODEL_URL` → `MODEL_HF_REPO`/`MODEL_HF_FILE`

> **Note:** `llama.cpp` requires GGUF format. Safetensors models must be converted first.

### Server tuning

| Variable | Description | Default |
|----------|-------------|---------|
| `LLAMA_CTX_SIZE` | Context window size (tokens) | `32768` (32K) |
| `LLAMA_N_GPU_LAYERS` | Number of layers offloaded to GPU | `999` (all) |
| `LLAMA_PARALLEL` | Number of parallel inference slots | `1` |
| `LLAMA_THREADS_HTTP` | HTTP server threads | `4` |
| `LLAMA_ENABLE_THINKING` | Enable hybrid thinking/reasoning mode | `false` |
| `LLAMA_EXTRA_ARGS` | Additional CLI flags for llama-server (space-separated) | — |
| `LLAMA_API_KEY` | API key for llama-server authentication | — |
| `LLAMA_DISABLE_WEBUI` | Disable the built-in web UI | `1` (disabled) |
| `LLAMA_DEFAULT_MAX_TOKENS` | `max_tokens` applied when a request omits one | — (uncapped) |

> **`LLAMA_DEFAULT_MAX_TOKENS`:** llama-server defaults to `n_predict = -1`, so a request that
> sets no limit generates until the context window is exhausted. If the model is ever loaded from a
> bad GGUF it emits garbage without an EOS token, and an uncapped request spends the full context
> producing it — minutes of output that look like a hang instead of an obviously wrong answer.
> Setting this (e.g. `1024`) bounds that failure to seconds. It only fills the gap: a request that
> supplies `max_tokens`, `max_completion_tokens`, or `n_predict` is left untouched, and it is only
> applied to completion endpoints.

> **Context window (`LLAMA_CTX_SIZE`):** The default is 32768 tokens (32K), which matches the native training context of Qwen3.6-27B. At 32K, the KV cache adds roughly 2–4 GB of VRAM on top of model weights (~19 GB for Q5_K_M), well within the RTX 5090's 32 GB. To override on RunPod without rebuilding the image, set `LLAMA_CTX_SIZE` in the **Environment Variables** section of your RunPod Endpoint or Template settings. Going beyond 32768 requires RoPE scaling and may degrade response quality.

> **Thinking mode:** When `LLAMA_ENABLE_THINKING=false` (the default), the server passes `--reasoning off` to suppress thinking token generation, and `reasoning_content` is stripped from API responses. Set to `true` to enable thinking/reasoning output. (`--reasoning off` replaced `--chat-template-kwargs '{"enable_thinking":false}'`, which newer llama-server builds warn is deprecated; it sets the same template kwarg plus the internal reasoning flag.)

### Timeouts

| Variable | Description | Default |
|----------|-------------|---------|
| `SERVER_START_TIMEOUT_SECONDS` | Max wait for llama-server health check | `900` |
| `SERVER_REQUEST_TIMEOUT_SECONDS` | Max wait for a single inference request | `600` |

### Internal (rarely changed)

| Variable | Description | Default |
|----------|-------------|---------|
| `LLAMA_SERVER_BIN` | Path to llama-server binary | `/app/llama-server` |
| `LLAMA_SERVER_HOST` | Bind address for llama-server | `127.0.0.1` |
| `LLAMA_SERVER_PORT` | Port for llama-server | `8080` |

## Project Structure

```
.
├── handler.py                  # RunPod worker: server lifecycle + request proxying
├── Dockerfile                  # CUDA-enabled image based on llama.cpp
├── requirements.txt            # Python dependencies (runpod, requests)
├── request-example.json        # Sample RunPod request body
├── runpod-endpoint-config.example.json  # Sample endpoint configuration
└── docs/
    └── diagrams/
        ├── architecture.svg    # System architecture
        └── data-flow.svg       # Request data flow
```

## Testing

There is no formal test suite. Use these verification steps:

```bash
# Syntax check
python3 -m py_compile handler.py

# Local container smoke test (requires NVIDIA GPU)
docker run --rm --gpus all \
  -e MODEL_URL=https://huggingface.co/marafx2025/Qwen3.6-27B-Abliterated-Heretic-Uncensored-GGUF/resolve/main/Qwen3.6-27B-Abliterated-Heretic-Uncensored-Q5_K_M.gguf \
  <your-image>:latest

# Send a test request to your RunPod endpoint
curl -X POST "https://api.runpod.ai/v2/<endpoint-id>/runsync" \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  -H "Content-Type: application/json" \
  -d @request-example.json
```

## Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit changes using imperative mood (`fix: handle missing MODEL_URL`)
4. Push to branch (`git push origin feature/my-feature`)
5. Open a Pull Request with purpose, config changes, and sample request/response

### Code style

- Python 3, PEP 8, 4-space indentation
- `snake_case` for functions/variables, `UPPER_SNAKE_CASE` for environment-driven constants
- Keep handler logic explicit and defensive

## License

See [LICENSE](LICENSE) for details.
