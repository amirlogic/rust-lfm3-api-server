# LFM3 API Server

An OpenAI-compatible HTTP API server that runs the **LFM3-1.7B** model locally using the **ONNX Runtime** via [`pykeio/ort`](https://github.com/pykeio/ort). Built with [Axum](https://github.com/tokio-rs/axum). Command-line interface only (no GUI required).

---

## Model Preparation

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

This creates the expected directory structure:

```text
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

## Building the Executable

### Production / Release Build (Recommended)
For model inference, use an optimized release build:

```powershell
cargo build --release
```

- **Output executable:**
  - Windows: `target\release\lfm3-server.exe`
  - Linux / macOS: `target/release/lfm3-server`

### Development / Debug Build
For fast compilation during development:

```powershell
cargo build
```

- **Output executable:**
  - Windows: `target\debug\lfm3-server.exe`
  - Linux / macOS: `target/debug/lfm3-server`

> [!NOTE]
> `ort` uses the **load-dynamic** feature, which automatically downloads or resolves the official ONNX Runtime shared library (`onnxruntime.dll` on Windows, `libonnxruntime.so` on Linux, `libonnxruntime.dylib` on macOS).
> If deploying the standalone executable to another machine, place `onnxruntime.dll` in the same directory as the executable, or set the `ORT_DYLIB_PATH` environment variable:
> ```powershell
> $env:ORT_DYLIB_PATH = "C:\path\to\onnxruntime.dll"
> ```

---

## Running the Server

### Direct Execution

```powershell
# Windows PowerShell
.\target\release\lfm3-server.exe --model-dir ./model --port 8080

# Linux / macOS
./target/release/lfm3-server --model-dir ./model --port 8080
```

### Using Environment Variables

```powershell
# Windows PowerShell
$env:LFM3_MODEL_DIR = "./model"
$env:LFM3_PORT = "8080"
.\target\release\lfm3-server.exe

# Bash
export LFM3_MODEL_DIR="./model"
export LFM3_PORT=8080
./target/release/lfm3-server
```

### CLI Options

View all available options:
```powershell
.\target\release\lfm3-server.exe --help
```

| Flag | Env Var | Default | Description |
|---|---|---|---|
| `-p`, `--port` | `LFM3_PORT` | `8080` | Port to listen on |
| `--host` | `LFM3_HOST` | `127.0.0.1` | Host address to bind |
| `-m`, `--model-dir` | `LFM3_MODEL_DIR` | *(Required)* | Path to directory containing model and tokenizer files |
| `--max-tokens` | `LFM3_MAX_TOKENS` | `512` | Default maximum tokens generated per completion |
| `-h`, `--help` | — | — | Print help information |
| `-V`, `--version` | — | — | Print version |

---

## API Endpoints

### 1. Health Check
```http
GET /health
```
**Response:**
```json
{
  "model": "lfm3",
  "status": "ok"
}
```

### 2. List Models
```http
GET /v1/models
```
**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "created": 0,
      "id": "lfm3",
      "object": "model",
      "owned_by": "local"
    }
  ]
}
```

### 3. Chat Completions (OpenAI-compatible)
```http
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "lfm3",
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user",   "content": "What is the capital of France?" }
  ],
  "max_tokens": 128
}
```

**Example with curl:**

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lfm3",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 128
  }'
```

> [!IMPORTANT]
> Streaming responses (`"stream": true`) are not supported. Set `"stream": false` or omit the field.

---

## Logging

Logging level is controlled via the `RUST_LOG` environment variable:

```powershell
# Windows PowerShell
$env:RUST_LOG = "debug"
.\target\release\lfm3-server.exe --model-dir ./model

# Bash
RUST_LOG=debug ./target/release/lfm3-server --model-dir ./model
```
