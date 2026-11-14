# LFM3 API Server

An OpenAI-compatible HTTP API server that runs the **LFM3-1.7B** model locally using the **ONNX Runtime** via [`pykeio/ort`](https://github.com/pykeio/ort). Built with [Axum](https://github.com/tokio-rs/axum). No GUI required.

---

## Model

Download the ONNX model from Hugging Face before running:

```bash
# Install huggingface-cli
pip install huggingface_hub[cli]

# Download the Q4 ONNX model (~1 GB)
huggingface-cli download onnx-community/LFM3-1.7B-ONNX \
  --include "config.json" "generation_config.json" "tokenizer.json" "tokenizer_config.json" \
              "onnx/model_q4.onnx" "onnx/model_q4.onnx_data" \
  --local-dir ./model
```

This creates the expected layout:

```
model/
  config.json
  generation_config.json
  tokenizer.json
  tokenizer_config.json
  onnx/
    model_q4.onnx
    model_q4.onnx_data
```

---

## Build

```bash
cargo build --release
```

> [!NOTE]
> `ort` uses the **load-dynamic** feature, which downloads the ONNX Runtime shared library automatically on first build (or you can set `ORT_DYLIB_PATH` to point to an existing install).

---

## Run

```bash
./target/release/lfm3-server --model-dir ./model --port 8080
```

| Flag / Env var | Default | Description |
|---|---|---|
| `--port` / `LFM3_PORT` | `8080` | HTTP port |
| `--host` / `LFM3_HOST` | `127.0.0.1` | Bind address |
| `--model-dir` / `LFM3_MODEL_DIR` | *(required)* | Path to model directory |
| `--max-tokens` / `LFM3_MAX_TOKENS` | `512` | Default max new tokens |

---

## API

### Health check

```
GET /health
```

### List models

```
GET /v1/models
```

### Chat completions (OpenAI-compatible)

```
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "lfm3",
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user",   "content": "What is the capital of France?" }
  ],
  "max_tokens": 256
}
```

**Example with curl:**

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lfm3",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 128
  }'
```

> [!IMPORTANT]
> Streaming (`"stream": true`) is not supported in this release. Set `"stream": false` or omit the field.

---

## Logging

Control verbosity via `RUST_LOG`:

```bash
RUST_LOG=debug ./target/release/lfm3-server --model-dir ./model
```
