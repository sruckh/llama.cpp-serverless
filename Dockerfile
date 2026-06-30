FROM ghcr.io/ggml-org/llama.cpp:server-cuda

WORKDIR /

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt
RUN pip3 install --break-system-packages --no-cache-dir -r /requirements.txt

COPY handler.py /handler.py

ENV PYTHONUNBUFFERED=1

# Discover every directory under /app that contains shared libraries
# and register them with the dynamic linker so llama-server can find
# libmtmd, libllama, libggml, etc. regardless of where the base image
# places them.
RUN find /app -name "*.so*" -exec dirname {} \; | sort -u \
      > /etc/ld.so.conf.d/llama.conf && ldconfig
ENV LLAMA_SERVER_BIN=/app/llama-server
ENV LLAMA_SERVER_HOST=127.0.0.1
ENV LLAMA_SERVER_PORT=8080
ENV LLAMA_CTX_SIZE=32768
ENV LLAMA_N_GPU_LAYERS=999
ENV LLAMA_THREADS_HTTP=4
ENV LLAMA_DISABLE_WEBUI=1

# Direct URL for the model GGUF — bypasses HF repo resolution.
# Override MODEL_URL / MMPROJ_URL on RunPod to swap models without rebuilding.
ENV MODEL_URL=https://huggingface.co/marafx2025/Qwen3.6-27B-Abliterated-Heretic-Uncensored-GGUF/resolve/main/Qwen3.6-27B-Abliterated-Heretic-Uncensored-Q5_K_M.gguf
ENV MMPROJ_URL=https://huggingface.co/marafx2025/Qwen3.6-27B-Abliterated-Heretic-Uncensored-GGUF/resolve/main/mmproj-F16.gguf

# The base image sets an entrypoint dispatcher that only recognises
# llama.cpp sub-commands (--server, --run, etc.).  Reset it so the
# container executes our Python handler directly.
ENTRYPOINT []
CMD ["python3", "-u", "/handler.py"]
