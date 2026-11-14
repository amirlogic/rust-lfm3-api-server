// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use axum::extract::{Json, State as AxumState};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Router, serve};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tauri::{Emitter, Manager, State};
use tauri_plugin_sql::{Migration, MigrationKind};
//use tauri::State;
//use tauri::{CustomMenuItem, Menu, MenuItem, Submenu};

#[tauri::command]
fn read_markdown_file(filename: &str) -> Result<String, String> {
    let mut file = File::open(filename).map_err(|e| e.to_string())?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .map_err(|e| e.to_string())?;
    Ok(contents)
}

#[tauri::command]
fn get_os() -> String {
    if cfg!(target_os = "windows") {
        "Windows".to_string()
    } else if cfg!(target_os = "linux") {
        "Linux".to_string()
    } else if cfg!(target_os = "macos") {
        "macOS".to_string()
    } else {
        "Unknown".to_string()
    }
}

fn model_files_exist(directory: &Path) -> bool {
    [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "onnx/model_q4.onnx",
        "onnx/model_q4.onnx_data",
    ]
    .iter()
    .all(|file| directory.join(file).is_file())
}

fn copy_directory(source: &Path, destination: &Path) -> io::Result<()> {
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let entry_type = entry.file_type()?;
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        if entry_type.is_dir() {
            copy_directory(&source_path, &destination_path)?;
        } else if entry_type.is_file() {
            fs::copy(source_path, destination_path)?;
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "The bundled model contains an unsupported file type",
            ));
        }
    }
    Ok(())
}

fn model_diagnostic_path(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    app.path()
        .app_local_data_dir()
        .map(|directory| directory.join("model-diagnostics.log"))
        .map_err(|error| error.to_string())
}

fn append_model_diagnostic(app: &tauri::AppHandle, message: &str) -> io::Result<()> {
    let path = model_diagnostic_path(app).map_err(io::Error::other)?;
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(file, "{message}")?;
    file.sync_data()
}

#[tauri::command]
fn begin_model_diagnostics(app: tauri::AppHandle) -> Result<String, String> {
    let path = model_diagnostic_path(&app)?;
    if let Some(directory) = path.parent() {
        fs::create_dir_all(directory).map_err(|error| error.to_string())?;
    }
    let mut file = File::create(&path).map_err(|error| error.to_string())?;
    writeln!(file, "Model diagnostics started").map_err(|error| error.to_string())?;
    file.sync_data().map_err(|error| error.to_string())?;
    Ok(path.to_string_lossy().into_owned())
}

#[tauri::command]
fn write_model_diagnostic(app: tauri::AppHandle, message: String) -> Result<(), String> {
    append_model_diagnostic(&app, &message).map_err(|error| error.to_string())
}

#[derive(Clone, Default)]
struct ApiServerState {
    address: Arc<Mutex<Option<String>>>,
    pending: Arc<
        Mutex<
            std::collections::HashMap<
                String,
                tokio::sync::oneshot::Sender<Result<String, String>>,
            >,
        >,
    >,
    next_request_id: Arc<AtomicU64>,
}

#[derive(Clone)]
struct ApiContext {
    app: tauri::AppHandle,
    state: ApiServerState,
}

#[derive(Clone, Deserialize, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatCompletionRequest {
    model: Option<String>,
    messages: Vec<ChatMessage>,
    max_tokens: Option<usize>,
    #[serde(default)]
    stream: bool,
}

fn unix_time_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

fn api_error(status: StatusCode, message: &str) -> Response {
    (
        status,
        Json(json!({ "error": { "message": message, "type": "invalid_request_error" } })),
    )
        .into_response()
}

async fn api_health() -> Json<Value> {
    Json(json!({ "status": "ok", "model": MODEL_ID }))
}

async fn api_models() -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{ "id": "lfm3", "object": "model", "owned_by": "local" }]
    }))
}

async fn chat_completions(
    AxumState(context): AxumState<ApiContext>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    if request
        .model
        .as_deref()
        .is_some_and(|model| model != "lfm3" && model != MODEL_ID)
    {
        return api_error(
            StatusCode::NOT_FOUND,
            "This server only provides the lfm3 model.",
        );
    }
    if request.stream {
        return api_error(
            StatusCode::BAD_REQUEST,
            "Streaming responses are not supported; set stream to false.",
        );
    }
    if request.messages.is_empty() {
        return api_error(StatusCode::BAD_REQUEST, "At least one message is required.");
    }
    if request.messages.iter().any(|message| {
        !matches!(message.role.as_str(), "system" | "user" | "assistant")
            || message.content.trim().is_empty()
    }) {
        return api_error(
            StatusCode::BAD_REQUEST,
            "Messages must have a supported role and non-empty text content.",
        );
    }
    let max_tokens = request.max_tokens.unwrap_or(512);
    if !(1..=4096).contains(&max_tokens) {
        return api_error(
            StatusCode::BAD_REQUEST,
            "max_tokens must be between 1 and 4096.",
        );
    }

    let id = format!(
        "chatcmpl-{}",
        context.state.next_request_id.fetch_add(1, Ordering::Relaxed) + 1
    );
    let started_at = unix_time_millis();
    let (response_tx, response_rx) = tokio::sync::oneshot::channel();
    {
        let mut pending = match context.state.pending.lock() {
            Ok(pending) => pending,
            Err(_) => {
                return api_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "API request state is unavailable.",
                )
            }
        };
        pending.insert(id.clone(), response_tx);
    }

    let event = json!({
        "id": id,
        "method": "POST",
        "path": "/v1/chat/completions",
        "status": "processing",
        "startedAt": started_at,
        "messages": request.messages,
        "maxTokens": max_tokens
    });
    if let Err(error) = context.app.emit("api-request", event) {
        if let Ok(mut pending) = context.state.pending.lock() {
            pending.remove(&id);
        }
        eprintln!("Could not notify the LFM3 window about an API request: {error}");
        return api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "The LFM3 inference window is unavailable.",
        );
    }

    let result = tokio::time::timeout(Duration::from_secs(300), response_rx).await;
    let elapsed = unix_time_millis().saturating_sub(started_at);
    let response_text = match result {
        Ok(Ok(Ok(text))) => text,
        Ok(Ok(Err(error))) => {
            emit_api_request_update(&context.app, &id, "failed", started_at, elapsed, None, Some(&error));
            return api_error(StatusCode::INTERNAL_SERVER_ERROR, &error);
        }
        Ok(Err(_)) => {
            let error = "The inference window closed before completing the request.";
            emit_api_request_update(&context.app, &id, "failed", started_at, elapsed, None, Some(error));
            return api_error(StatusCode::SERVICE_UNAVAILABLE, error);
        }
        Err(_) => {
            if let Ok(mut pending) = context.state.pending.lock() {
                pending.remove(&id);
            }
            let error = "The inference request timed out.";
            emit_api_request_update(&context.app, &id, "failed", started_at, elapsed, None, Some(error));
            return api_error(StatusCode::GATEWAY_TIMEOUT, error);
        }
    };

    emit_api_request_update(
        &context.app,
        &id,
        "completed",
        started_at,
        elapsed,
        Some(&response_text),
        None,
    );
    Json(json!({
        "id": id,
        "object": "chat.completion",
        "created": started_at / 1000,
        "model": "lfm3",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": response_text },
            "finish_reason": "stop"
        }]
    }))
    .into_response()
}

fn emit_api_request_update(
    app: &tauri::AppHandle,
    id: &str,
    status: &str,
    started_at: u128,
    duration_ms: u128,
    response: Option<&str>,
    error: Option<&str>,
) {
    if let Err(emit_error) = app.emit(
        "api-request",
        json!({
            "id": id,
            "method": "POST",
            "path": "/v1/chat/completions",
            "status": status,
            "startedAt": started_at,
            "durationMs": duration_ms,
            "response": response,
            "error": error
        }),
    ) {
        eprintln!("Could not update the LFM3 request monitor: {emit_error}");
    }
}

#[tauri::command]
async fn start_api_server(
    app: tauri::AppHandle,
    state: State<'_, ApiServerState>,
) -> Result<String, String> {
    {
        let address = state
            .address
            .lock()
            .map_err(|_| "API server state is unavailable".to_string())?;
        if let Some(address) = address.as_ref() {
            return Ok(address.clone());
        }
    }

    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .map_err(|error| format!("Could not bind the local API server: {error}"))?;
    let socket_address = listener
        .local_addr()
        .map_err(|error| format!("Could not read the local API address: {error}"))?;
    let url = format!("http://{socket_address}");
    let mut address = state
        .address
        .lock()
        .map_err(|_| "API server state is unavailable".to_string())?;
    if let Some(address) = address.as_ref() {
        return Ok(address.clone());
    }
    let context = ApiContext {
        app: app.clone(),
        state: state.inner().clone(),
    };
    let router = Router::new()
        .route("/health", get(api_health))
        .route("/v1/models", get(api_models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(context);

    *address = Some(url.clone());
    drop(address);
    tauri::async_runtime::spawn(async move {
        if let Err(error) = serve(listener, router).await {
            eprintln!("LFM3 API server stopped: {error}");
            let _ = app.emit("api-server-error", error.to_string());
        }
    });
    Ok(url)
}

#[tauri::command]
fn resolve_api_request(
    state: State<'_, ApiServerState>,
    request_id: String,
    response: Option<String>,
    error: Option<String>,
) -> Result<(), String> {
    if response.is_some() == error.is_some() {
        return Err("Provide exactly one of response or error.".to_string());
    }
    let sender = state
        .pending
        .lock()
        .map_err(|_| "API request state is unavailable".to_string())?
        .remove(&request_id)
        .ok_or_else(|| format!("No pending API request has ID {request_id}"))?;
    sender
        .send(match (response, error) {
            (Some(response), None) => Ok(response),
            (None, Some(error)) => Err(error),
            _ => return Err("Provide exactly one of response or error.".to_string()),
        })
        .map_err(|_| "The HTTP request ended before inference completed.".to_string())
}

const MODEL_ID: &str = "onnx-community/LFM3-1.7B-ONNX";
const MODEL_FILES: &[&str] = &[
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "onnx/model_q4.onnx",
    "onnx/model_q4.onnx_data",
];

fn start_model_server(app: &tauri::AppHandle, model_directory: PathBuf) -> Result<String, String> {
    static SERVER_URL: OnceLock<String> = OnceLock::new();
    if let Some(url) = SERVER_URL.get() {
        return Ok(url.clone());
    }

    let window = app
        .get_webview_window("main")
        .ok_or_else(|| "Could not find the main window for the model server".to_string())?;
    let origin = window
        .url()
        .map_err(|error| error.to_string())?
        .origin()
        .ascii_serialization();
    let listener = TcpListener::bind(("127.0.0.1", 0)).map_err(|error| error.to_string())?;
    let address = listener.local_addr().map_err(|error| error.to_string())?;
    let base_url = format!("http://{address}/");
    let diagnostic_path = model_diagnostic_path(app)?;

    std::thread::Builder::new()
        .name("model-http-server".to_string())
        .spawn(move || {
            for connection in listener.incoming() {
                match connection {
                    Ok(stream) => {
                        let model_directory = model_directory.clone();
                        let origin = origin.clone();
                        let diagnostic_path = diagnostic_path.clone();
                        if let Err(error) = std::thread::Builder::new()
                            .name("model-http-request".to_string())
                            .spawn(move || {
                                if let Err(error) = serve_model_request(
                                    stream,
                                    &model_directory,
                                    &origin,
                                    &diagnostic_path,
                                ) {
                                    eprintln!("Model HTTP request failed: {error}");
                                    log_model_server(
                                        &diagnostic_path,
                                        &format!("Request error: {error}"),
                                    );
                                }
                            })
                        {
                            eprintln!("Could not start model HTTP request thread: {error}");
                        }
                    }
                    Err(error) => {
                        eprintln!("Model HTTP server accept failed: {error}");
                        log_model_server(&diagnostic_path, &format!("Accept error: {error}"));
                    }
                }
            }
        })
        .map_err(|error| error.to_string())?;

    SERVER_URL
        .set(base_url.clone())
        .map_err(|_| "The model HTTP server was started more than once".to_string())?;
    append_model_diagnostic(app, &format!("Model HTTP server listening at {base_url}"))
        .map_err(|error| error.to_string())?;
    Ok(base_url)
}

fn log_model_server(path: &Path, message: &str) {
    let result = (|| -> io::Result<()> {
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        writeln!(file, "{message}")?;
        file.sync_data()
    })();
    if let Err(error) = result {
        eprintln!("Could not write model server diagnostic: {error}");
    }
}

fn respond(
    stream: &mut TcpStream,
    status: &str,
    headers: &[(&str, String)],
    body: &[u8],
) -> io::Result<()> {
    write!(stream, "HTTP/1.1 {status}\r\nConnection: close\r\n")?;
    for (name, value) in headers {
        write!(stream, "{name}: {value}\r\n")?;
    }
    write!(stream, "Content-Length: {}\r\n\r\n", body.len())?;
    stream.write_all(body)
}

fn serve_model_request(
    stream: TcpStream,
    model_directory: &Path,
    allowed_origin: &str,
    diagnostic_path: &Path,
) -> io::Result<()> {
    const MAX_HEADER_BYTES: usize = 16 * 1024;
    stream.set_read_timeout(Some(std::time::Duration::from_secs(10)))?;
    let mut reader = BufReader::new(stream);
    let mut request_line = String::new();
    reader.read_line(&mut request_line)?;
    if request_line.len() > MAX_HEADER_BYTES {
        return respond(
            reader.get_mut(),
            "431 Request Header Fields Too Large",
            &[],
            b"",
        );
    }
    let mut request_parts = request_line.split_whitespace();
    let method = request_parts.next().unwrap_or("");
    let target = request_parts.next().unwrap_or("");
    let mut origin = None;
    let mut range = None;
    let mut total_header_bytes = request_line.len();
    loop {
        let mut line = String::new();
        let bytes = reader.read_line(&mut line)?;
        total_header_bytes += bytes;
        if total_header_bytes > MAX_HEADER_BYTES {
            return respond(
                reader.get_mut(),
                "431 Request Header Fields Too Large",
                &[],
                b"",
            );
        }
        if bytes == 0 || line == "\r\n" || line == "\n" {
            break;
        }
        if let Some((name, value)) = line.split_once(':') {
            match name.trim().to_ascii_lowercase().as_str() {
                "origin" => origin = Some(value.trim().to_string()),
                "range" => range = Some(value.trim().to_string()),
                _ => {}
            }
        }
    }

    if origin.as_deref() != Some(allowed_origin) {
        log_model_server(
            diagnostic_path,
            &format!("Rejected request with origin {origin:?}"),
        );
        return respond(reader.get_mut(), "403 Forbidden", &[], b"");
    }

    let cors_headers = [
        ("Access-Control-Allow-Origin", allowed_origin.to_string()),
        (
            "Access-Control-Allow-Methods",
            "GET, HEAD, OPTIONS".to_string(),
        ),
        ("Access-Control-Allow-Headers", "Range".to_string()),
        (
            "Access-Control-Expose-Headers",
            "Accept-Ranges, Content-Range".to_string(),
        ),
        ("Vary", "Origin".to_string()),
    ];
    if method == "OPTIONS" {
        return respond(reader.get_mut(), "204 No Content", &cors_headers, b"");
    }
    if method != "GET" && method != "HEAD" {
        return respond(
            reader.get_mut(),
            "405 Method Not Allowed",
            &cors_headers,
            b"",
        );
    }

    let request_path = target.split('?').next().unwrap_or("");
    let relative_path = MODEL_FILES
        .iter()
        .find(|file| request_path == format!("/{MODEL_ID}/{file}"));
    let Some(relative_path) = relative_path else {
        log_model_server(diagnostic_path, &format!("Not found: {request_path}"));
        return respond(reader.get_mut(), "404 Not Found", &cors_headers, b"");
    };
    let file_path = model_directory.join(MODEL_ID).join(relative_path);
    let mut file = match File::open(&file_path) {
        Ok(file) => file,
        Err(error) => {
            log_model_server(
                diagnostic_path,
                &format!("Could not open {}: {error}", file_path.display()),
            );
            return respond(reader.get_mut(), "404 Not Found", &cors_headers, b"");
        }
    };
    let file_size = file.metadata()?.len();
    let mut status = "200 OK";
    let mut start = 0;
    let mut length = file_size;
    let mut response_headers = cors_headers.to_vec();
    response_headers.push(("Accept-Ranges", "bytes".to_string()));
    response_headers.push((
        "Content-Type",
        if relative_path.ends_with(".json") {
            "application/json".to_string()
        } else {
            "application/octet-stream".to_string()
        },
    ));

    if let Some(range) = range {
        match parse_byte_range(&range, file_size) {
            Some((range_start, range_end)) => {
                start = range_start;
                length = range_end - range_start + 1;
                status = "206 Partial Content";
                response_headers.push((
                    "Content-Range",
                    format!("bytes {range_start}-{range_end}/{file_size}"),
                ));
            }
            None => {
                response_headers.push(("Content-Range", format!("bytes */{file_size}")));
                return respond(
                    reader.get_mut(),
                    "416 Range Not Satisfiable",
                    &response_headers,
                    b"",
                );
            }
        }
    }

    write!(
        reader.get_mut(),
        "HTTP/1.1 {status}\r\nConnection: close\r\n"
    )?;
    for (name, value) in &response_headers {
        write!(reader.get_mut(), "{name}: {value}\r\n")?;
    }
    write!(reader.get_mut(), "Content-Length: {length}\r\n\r\n")?;
    if method != "HEAD" {
        file.seek(SeekFrom::Start(start))?;
        let copied = io::copy(&mut file.take(length), reader.get_mut())?;
        if copied != length {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "Served {copied} of {length} bytes from {}",
                    file_path.display()
                ),
            ));
        }
    }
    log_model_server(
        diagnostic_path,
        &format!("Served {request_path}: {length} of {file_size} bytes ({status})"),
    );
    Ok(())
}

fn parse_byte_range(range: &str, file_size: u64) -> Option<(u64, u64)> {
    if file_size == 0 {
        return None;
    }
    let range = range.strip_prefix("bytes=")?;
    let (start, end) = range.split_once('-')?;
    if start.is_empty() {
        let suffix_length = end.parse::<u64>().ok()?.min(file_size);
        return (suffix_length > 0).then_some((file_size - suffix_length, file_size - 1));
    }
    let start = start.parse::<u64>().ok()?;
    if start >= file_size {
        return None;
    }
    let end = if end.is_empty() {
        file_size - 1
    } else {
        end.parse::<u64>().ok()?.min(file_size - 1)
    };
    (end >= start).then_some((start, end))
}

#[tauri::command]
fn prepare_model(app: tauri::AppHandle) -> Result<String, String> {
    let resource_root = if cfg!(debug_assertions) {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("resources")
    } else {
        app.path()
            .resource_dir()
            .map_err(|error| error.to_string())?
    };
    let model_relative_path = Path::new("onnx-community").join("LFM3-1.7B-ONNX");
    let source = resource_root.join("models").join(&model_relative_path);
    if let Err(error) = append_model_diagnostic(
        &app,
        &format!("Preparing model: source={}", source.display()),
    ) {
        eprintln!("Could not write model diagnostic: {error}");
    }
    if !model_files_exist(&source) {
        return Err(format!(
            "The bundled LFM3 model is missing or incomplete at {}",
            source.display()
        ));
    }

    let model_directory = app
        .path()
        .app_local_data_dir()
        .map_err(|error| error.to_string())?
        .join("models");
    let destination = model_directory.join(&model_relative_path);
    if let Err(error) = append_model_diagnostic(
        &app,
        &format!("Model destination: {}", destination.display()),
    ) {
        eprintln!("Could not write model diagnostic: {error}");
    }
    if !model_files_exist(&destination) {
        if let Err(error) = append_model_diagnostic(&app, "Extracting bundled model") {
            eprintln!("Could not write model diagnostic: {error}");
        }
        copy_directory(&resource_root.join("models"), &model_directory)
            .map_err(|error| format!("Could not extract the bundled LFM3 model: {error}"))?;
        if let Err(error) = append_model_diagnostic(&app, "Model extraction complete") {
            eprintln!("Could not write model diagnostic: {error}");
        }
    }

    for file in [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "onnx/model_q4.onnx",
        "onnx/model_q4.onnx_data",
    ] {
        let file_path = destination.join(file);
        let size = fs::metadata(&file_path)
            .map(|metadata| metadata.len().to_string())
            .unwrap_or_else(|error| format!("unavailable ({error})"));
        if let Err(error) =
            append_model_diagnostic(&app, &format!("Model file: {file} ({size} bytes)"))
        {
            eprintln!("Could not write model diagnostic: {error}");
            break;
        }
    }

    start_model_server(&app, model_directory)
}

fn main() {
    let migrations = vec![
        Migration {
            version: 1,
            description: "create_image_embedding_cache",
            sql: "CREATE TABLE image_embeddings (
                root_path TEXT NOT NULL,
                model_id TEXT NOT NULL,
                model_revision TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                modified_ms INTEGER NOT NULL,
                embedding TEXT NOT NULL,
                PRIMARY KEY (root_path, model_id, model_revision, relative_path)
            );",
            kind: MigrationKind::Up,
        },
        Migration {
            version: 2,
            description: "persist_last_search_per_collection",
            sql: "CREATE TABLE image_search_state (
                root_path TEXT NOT NULL,
                model_id TEXT NOT NULL,
                model_revision TEXT NOT NULL,
                query TEXT NOT NULL,
                similarity_threshold REAL NOT NULL,
                query_embedding TEXT NOT NULL,
                PRIMARY KEY (root_path, model_id, model_revision)
            );",
            kind: MigrationKind::Up,
        },
    ];

    tauri::Builder::default()
        .manage(ApiServerState::default())
        .plugin(tauri_plugin_cli::init())
        .plugin(tauri_plugin_os::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_store::Builder::new().build())
        .plugin(tauri_plugin_fs::init())
        .plugin(
            tauri_plugin_sql::Builder::default()
                .add_migrations("sqlite:image-embeddings.sqlite3", migrations)
                .build(),
        )
        .invoke_handler(tauri::generate_handler![
            read_markdown_file,
            get_os,
            prepare_model,
            begin_model_diagnostics,
            write_model_diagnostic,
            start_api_server,
            resolve_api_request
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

/* use std::fs::File;
use std::io::{self, Read};

fn read_markdown_file(filename: &str) -> io::Result<String> {
    let mut file = File::open(filename)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
}

#[cfg(target_os = "windows")]
fn get_os() -> &'static str {
    "Windows"
}

#[cfg(target_os = "linux")]
fn get_os() -> &'static str {
    "Linux"
}

#[cfg(target_os = "macos")]
fn get_os() -> &'static str {
    "macOS"
} */

/* fn main() {
    tauri2_lib::run()
}
 */
