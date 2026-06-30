import atexit
import os
import subprocess
import time
from typing import Any, Dict, List

import requests
import runpod


LLAMA_SERVER_BIN = os.getenv("LLAMA_SERVER_BIN", "/app/llama-server")
LLAMA_SERVER_HOST = os.getenv("LLAMA_SERVER_HOST", "127.0.0.1")
LLAMA_SERVER_PORT = int(os.getenv("LLAMA_SERVER_PORT", "8080"))
LLAMA_SERVER_URL = f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}"

SERVER_START_TIMEOUT_SECONDS = int(os.getenv("SERVER_START_TIMEOUT_SECONDS", "900"))
SERVER_REQUEST_TIMEOUT_SECONDS = int(os.getenv("SERVER_REQUEST_TIMEOUT_SECONDS", "600"))

_server_process = None


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _build_server_command() -> List[str]:
    cmd = [
        LLAMA_SERVER_BIN,
        "--host",
        LLAMA_SERVER_HOST,
        "--port",
        str(LLAMA_SERVER_PORT),
        "--ctx-size",
        os.getenv("LLAMA_CTX_SIZE", "32768"),
        "--n-gpu-layers",
        os.getenv("LLAMA_N_GPU_LAYERS", "999"),
        "--threads-http",
        os.getenv("LLAMA_THREADS_HTTP", "4"),
        "--parallel",
        os.getenv("LLAMA_PARALLEL", "1"),
    ]

    if _env_flag("LLAMA_DISABLE_WEBUI", True):
        cmd.append("--no-webui")

    model_path = os.getenv("MODEL_PATH", "").strip()
    hf_repo = os.getenv("MODEL_HF_REPO", "").strip()
    hf_file = os.getenv("MODEL_HF_FILE", "").strip()

    model_url = os.getenv("MODEL_URL", "").strip()

    if model_path:
        cmd.extend(["--model", model_path])
    elif model_url:
        cmd.extend(["--model-url", model_url])
    elif hf_repo:
        cmd.extend(["--hf-repo", hf_repo])
        if hf_file:
            cmd.extend(["--hf-file", hf_file])
    else:
        raise RuntimeError(
            "No model configured. Set MODEL_PATH, MODEL_URL, or MODEL_HF_REPO (and optionally MODEL_HF_FILE)."
        )

    mmproj_path = os.getenv("MMPROJ_PATH", "").strip()
    mmproj_url = os.getenv("MMPROJ_URL", "").strip()

    if mmproj_path:
        cmd.extend(["--mmproj", mmproj_path])
    elif mmproj_url:
        cmd.extend(["--mmproj-url", mmproj_url])
    else:
        cmd.append("--no-mmproj")

    if not _env_flag("LLAMA_ENABLE_THINKING", False):
        cmd.extend(["--chat-template-kwargs", '{"enable_thinking":false}'])

    alias = os.getenv("MODEL_ALIAS", "gpt-4o").strip()
    if alias:
        cmd.extend(["--alias", alias])

    api_key = os.getenv("LLAMA_API_KEY", "").strip()
    if api_key:
        cmd.extend(["--api-key", api_key])

    extra_args = os.getenv("LLAMA_EXTRA_ARGS", "").strip()
    if extra_args:
        cmd.extend(extra_args.split())

    return cmd


def _wait_for_server_ready() -> None:
    deadline = time.time() + SERVER_START_TIMEOUT_SECONDS
    health_url = f"{LLAMA_SERVER_URL}/health"

    while time.time() < deadline:
        if _server_process is not None and _server_process.poll() is not None:
            output = _server_process.stdout.read().decode(errors="replace")[-2000:]
            raise RuntimeError(
                f"llama-server exited with code {_server_process.returncode}: {output}"
            )

        try:
            response = requests.get(health_url, timeout=10)
            if response.ok:
                return
        except requests.RequestException:
            pass

        time.sleep(2)

    raise TimeoutError(
        f"llama-server did not become ready within {SERVER_START_TIMEOUT_SECONDS} seconds"
    )


def _ensure_server_running() -> None:
    global _server_process

    if _server_process is not None and _server_process.poll() is None:
        return

    command = _build_server_command()
    _server_process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    _wait_for_server_ready()


def _stop_server() -> None:
    global _server_process

    if _server_process is None:
        return

    if _server_process.poll() is None:
        _server_process.terminate()
        try:
            _server_process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            _server_process.kill()

    _server_process = None


def _build_default_chat_payload(job_input: Dict[str, Any]) -> Dict[str, Any]:
    messages = job_input.get("messages")
    if messages is None:
        prompt = job_input.get("prompt", "Hello")
        messages = [{"role": "user", "content": prompt}]

    payload: Dict[str, Any] = {
        "model": job_input.get("model", os.getenv("MODEL_ALIAS", "gpt-4o").strip()),
        "messages": messages,
        "stream": False,
    }

    optional_keys = [
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "presence_penalty",
        "frequency_penalty",
        "stop",
        "response_format",
        "tools",
        "tool_choice",
    ]

    for key in optional_keys:
        if key in job_input:
            payload[key] = job_input[key]

    if "params" in job_input and isinstance(job_input["params"], dict):
        payload.update(job_input["params"])

    return payload


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        _ensure_server_running()
    except Exception as exc:
        return {
            "error": f"Failed to initialize llama-server: {exc}",
            "refresh_worker": True,
        }

    job_input = job.get("input", {})

    if job_input.get("debug"):
        try:
            models_response = requests.get(f"{LLAMA_SERVER_URL}/v1/models", timeout=10)
            models_body = models_response.json()
        except Exception as exc:
            models_body = {"error": str(exc)}

        return {
            "ok": True,
            "debug": True,
            "server_command": _build_server_command(),
            "model_alias_env_raw": os.getenv("MODEL_ALIAS"),
            "models_endpoint": models_body,
        }

    endpoint = job_input.get("endpoint", "/v1/chat/completions")
    payload = job_input.get("payload")

    if payload is None:
        payload = _build_default_chat_payload(job_input)

    # Always normalize model name to the alias llama-server was started with
    payload["model"] = os.getenv("MODEL_ALIAS", "gpt-4o").strip()

    url = f"{LLAMA_SERVER_URL}{endpoint}"

    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("LLAMA_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=SERVER_REQUEST_TIMEOUT_SECONDS,
        )

        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type.lower():
            body = response.json()
            if not _env_flag("LLAMA_ENABLE_THINKING", False) and isinstance(body, dict):
                for choice in body.get("choices", []):
                    msg = choice.get("message", {})
                    msg.pop("reasoning_content", None)
        else:
            body = {"raw": response.text}

        result = {
            "status_code": response.status_code,
            "ok": response.ok,
            "endpoint": endpoint,
            "result": body,
        }

        if not response.ok:
            result["refresh_worker"] = False

        return result
    except Exception as exc:
        return {
            "error": f"Inference request failed: {exc}",
            "refresh_worker": False,
        }


atexit.register(_stop_server)

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
