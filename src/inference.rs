//! Inference engine — loads the LFM3 ONNX model and runs autoregressive generation.
//!
//! Architecture notes
//! ------------------
//! LFM3 is a tri-hybrid model (gated short-conv + sliding-window attention + global attention).
//! The ONNX export from `onnx-community/LFM3-1.7B-ONNX` bundles both the KV-attention caches
//! (`past_key_values.N.key` / `past_key_values.N.value`) and the convolution states
//! (`past_conv_states.N`) as dynamic inputs / outputs so the generation loop can be run
//! entirely inside ORT without touching Python.
//!
//! The number of cache layers is discovered dynamically by inspecting the session's input metadata
//! at startup.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{bail, Context};
use ndarray::{Array2, ArrayD, Axis, IxDyn};
use ort::{session::Session, value::Tensor};
use tokenizers::Tokenizer;
use tracing::{debug, info, instrument};

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// LFM3 / LLaMA-3 EOS token ID
const EOS_TOKEN_ID: i64 = 128_001;

// ─────────────────────────────────────────────────────────────────────────────
// Public interface
// ─────────────────────────────────────────────────────────────────────────────

/// A loaded, thread-safe inference engine.
#[derive(Clone)]
pub struct Engine(Arc<Mutex<EngineInner>>);

impl Engine {
    /// Load model + tokenizer from `model_dir`.
    ///
    /// Expected layout:
    /// ```text
    /// model_dir/
    ///   tokenizer.json
    ///   onnx/
    ///     model_q4.onnx          (+ model_q4.onnx_data alongside)
    /// ```
    pub fn load(model_dir: &Path, default_max_tokens: usize) -> anyhow::Result<Self> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        let onnx_path = model_dir.join("onnx").join("model_q4.onnx");

        // --- Tokenizer --------------------------------------------------
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from {}: {e}", tokenizer_path.display()))?;

        // --- ONNX Runtime -----------------------------------------------
        let env = ort::init()
            .with_name("lfm3")
            .build()
            .context("Failed to initialise ONNX Runtime environment")?;

        let session = Session::builder(&env)
            .context("Failed to create ORT SessionBuilder")?
            .commit_from_file(&onnx_path)
            .with_context(|| format!("Failed to load ONNX model from {}", onnx_path.display()))?;

        // --- Discover cache tensor layout --------------------------------
        let cache_layout = discover_cache_layout(&session)?;
        info!(
            kv_layers = cache_layout.num_kv_layers,
            conv_layers = cache_layout.num_conv_layers,
            has_position_ids = cache_layout.has_position_ids,
            "Cache layout discovered"
        );

        Ok(Self(Arc::new(Mutex::new(EngineInner {
            session,
            tokenizer,
            cache_layout,
            default_max_tokens,
        }))))
    }

    /// Generate a response for a chat conversation.
    pub fn generate(
        &self,
        messages: &[crate::server::ChatMessage],
        max_tokens: usize,
    ) -> anyhow::Result<GenerateResult> {
        let mut inner = self.0.lock().map_err(|_| anyhow::anyhow!("Engine mutex poisoned"))?;
        inner.generate(messages, max_tokens)
    }

    pub fn default_max_tokens(&self) -> usize {
        self.0.lock().map(|i| i.default_max_tokens).unwrap_or(512)
    }
}

pub struct GenerateResult {
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Cache layout discovery
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct CacheLayout {
    num_kv_layers: usize,
    num_conv_layers: usize,
    /// True when the model expects `position_ids` as an explicit input.
    has_position_ids: bool,
}

fn discover_cache_layout(session: &Session) -> anyhow::Result<CacheLayout> {
    let input_names: Vec<String> = session
        .inputs()
        .iter()
        .map(|outlet| outlet.name().to_string())
        .collect();

    let num_kv_layers = input_names
        .iter()
        .filter(|n: &&String| n.starts_with("past_key_values.") && n.ends_with(".key"))
        .count();

    let num_conv_layers = input_names
        .iter()
        .filter(|n: &&String| n.starts_with("past_conv_states."))
        .count();

    let has_position_ids = input_names.iter().any(|n: &String| n == "position_ids");

    Ok(CacheLayout {
        num_kv_layers,
        num_conv_layers,
        has_position_ids,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Inner engine
// ─────────────────────────────────────────────────────────────────────────────

struct EngineInner {
    session: Session,
    tokenizer: Tokenizer,
    cache_layout: CacheLayout,
    default_max_tokens: usize,
}

impl EngineInner {
    #[instrument(skip(self, messages), fields(max_tokens))]
    fn generate(
        &mut self,
        messages: &[crate::server::ChatMessage],
        max_tokens: usize,
    ) -> anyhow::Result<GenerateResult> {
        // 1. Build prompt text using LFM3 / LLaMA-3 chat template
        let prompt = build_chat_prompt(messages);
        debug!(prompt_len = prompt.len(), "Built chat prompt");

        // 2. Tokenise
        let encoding = self
            .tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|e| anyhow::anyhow!("Tokenisation failed: {e}"))?;
        let prompt_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let prompt_tokens = prompt_ids.len();

        if prompt_tokens == 0 {
            bail!("Empty prompt after tokenisation");
        }

        debug!(prompt_tokens, "Prompt tokenised");

        // 3. Autoregressive generation
        let mut all_ids = prompt_ids.clone();
        let mut generated: Vec<i64> = Vec::new();

        // KV + conv caches — will be populated after the first inference step
        let mut kv_cache: Option<KvCache> = None;

        for step in 0..max_tokens {
            // First step: full prompt; subsequent steps: only the last generated token
            let input_ids_slice: Vec<i64> = if step == 0 {
                all_ids.clone()
            } else {
                vec![*all_ids.last().unwrap()]
            };
            let seq_len = input_ids_slice.len();
            let total_len = all_ids.len();

            // ── Build owned arrays ──────────────────────────────────────────
            let input_ids_arr =
                Array2::from_shape_vec((1, seq_len), input_ids_slice)
                    .context("Failed to build input_ids array")?;

            let attn_mask_arr = Array2::from_elem((1, total_len), 1i64);

            let position_ids_arr = Array2::from_shape_vec(
                (1, seq_len),
                ((total_len - seq_len) as i64..total_len as i64).collect::<Vec<_>>(),
            )
            .context("Failed to build position_ids array")?;

            // ── Assemble named inputs using ort::inputs! macro ─────────────
            // The macro produces a Vec<(Cow<str>, SessionInputValue)>.
            // We start with the fixed inputs then push cache tensors.
            let mut inputs = if self.cache_layout.has_position_ids {
                ort::inputs![
                    "input_ids"      => Tensor::from_array(input_ids_arr)?,
                    "attention_mask" => Tensor::from_array(attn_mask_arr)?,
                    "position_ids"   => Tensor::from_array(position_ids_arr)?,
                ]
            } else {
                ort::inputs![
                    "input_ids"      => Tensor::from_array(input_ids_arr)?,
                    "attention_mask" => Tensor::from_array(attn_mask_arr)?,
                ]
            };

            // ── Cache tensors ───────────────────────────────────────────────
            if let Some(ref cache) = kv_cache {
                for (layer_idx, (k, v)) in cache.kv.iter().enumerate() {
                    inputs.push((
                        format!("past_key_values.{layer_idx}.key").into(),
                        Tensor::from_array(k.clone())?.into(),
                    ));
                    inputs.push((
                        format!("past_key_values.{layer_idx}.value").into(),
                        Tensor::from_array(v.clone())?.into(),
                    ));
                }
                for (layer_idx, s) in cache.conv.iter().enumerate() {
                    inputs.push((
                        format!("past_conv_states.{layer_idx}").into(),
                        Tensor::from_array(s.clone())?.into(),
                    ));
                }
            } else {
                // Step 0 — provide empty zero-element caches for each expected input
                let all_input_names: Vec<String> = self
                    .session
                    .inputs()
                    .iter()
                    .map(|o| o.name().to_string())
                    .collect();

                for name in &all_input_names {
                    if (name.starts_with("past_key_values.") || name.starts_with("past_conv_states."))
                        && !inputs.iter().any(|(n, _)| n.as_ref() == name.as_str())
                    {
                        // Empty 4-D tensor [1, 1, 0, 1] — ORT treats the 0-length dim as "no cache yet"
                        let empty: ArrayD<f32> = ArrayD::zeros(IxDyn(&[1, 1, 0, 1]));
                        inputs.push((
                            name.clone().into(),
                            Tensor::from_array(empty)?.into(),
                        ));
                    }
                }
            }

            // ── Run the session ─────────────────────────────────────────────
            let outputs = self
                .session
                .run(inputs)
                .context("ONNX Runtime inference failed")?;

            // ── Extract logits — shape [1, seq_len, vocab_size] ────────────
            let logits_val = outputs
                .get("logits")
                .context("Model did not return 'logits' output")?;

            let logits_view = logits_val
                .try_extract_array::<f32>()
                .context("Failed to extract logits as f32 array")?;

            let last_pos = logits_view.shape()[1] - 1;
            let last_logits = logits_view.index_axis(Axis(1), last_pos);

            // Greedy argmax
            let next_token = last_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as i64)
                .unwrap_or(0);

            debug!(step, next_token, "Generated token");

            all_ids.push(next_token);
            generated.push(next_token);

            // Update KV / conv cache from outputs
            kv_cache = Some(extract_kv_cache(&outputs, &self.cache_layout)?);

            // Stop on EOS
            if next_token == EOS_TOKEN_ID {
                debug!("EOS token reached at step {step}");
                break;
            }
        }

        // 4. Decode generated tokens
        let gen_u32: Vec<u32> = generated.iter().map(|&t| t as u32).collect();
        let text = self
            .tokenizer
            .decode(&gen_u32, true)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {e}"))?;

        Ok(GenerateResult {
            text,
            prompt_tokens,
            completion_tokens: generated.len(),
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// KV cache helpers
// ─────────────────────────────────────────────────────────────────────────────

struct KvCache {
    kv: Vec<(ArrayD<f32>, ArrayD<f32>)>,
    conv: Vec<ArrayD<f32>>,
}

fn extract_kv_cache(
    outputs: &ort::session::SessionOutputs<'_>,
    layout: &CacheLayout,
) -> anyhow::Result<KvCache> {
    let mut kv: Vec<(ArrayD<f32>, ArrayD<f32>)> = Vec::new();
    let mut conv: Vec<ArrayD<f32>> = Vec::new();

    for i in 0..layout.num_kv_layers {
        let key = extract_f32_array(outputs, &format!("present.{i}.key"))
            .or_else(|_| extract_f32_array(outputs, &format!("present_key_values.{i}.key")))
            .with_context(|| format!("Missing present key for layer {i}"))?;
        let val = extract_f32_array(outputs, &format!("present.{i}.value"))
            .or_else(|_| extract_f32_array(outputs, &format!("present_key_values.{i}.value")))
            .with_context(|| format!("Missing present value for layer {i}"))?;
        kv.push((key, val));
    }

    for i in 0..layout.num_conv_layers {
        let s = extract_f32_array(outputs, &format!("present_conv_states.{i}"))
            .with_context(|| format!("Missing present conv state for layer {i}"))?;
        conv.push(s);
    }

    Ok(KvCache { kv, conv })
}

fn extract_f32_array(
    outputs: &ort::session::SessionOutputs<'_>,
    name: &str,
) -> anyhow::Result<ArrayD<f32>> {
    let val = outputs
        .get(name)
        .with_context(|| format!("Output '{name}' not found"))?;
    let view = val
        .try_extract_array::<f32>()
        .with_context(|| format!("Could not extract f32 array for '{name}'"))?;
    Ok(view.to_owned())
}

// ─────────────────────────────────────────────────────────────────────────────
// Chat prompt builder
// ─────────────────────────────────────────────────────────────────────────────

/// Build a prompt string from a list of messages using LFM3 / LLaMA-3 chat template.
fn build_chat_prompt(messages: &[crate::server::ChatMessage]) -> String {
    let mut prompt = String::from("<|begin_of_text|>");
    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                prompt.push_str(&format!(
                    "<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>",
                    msg.content.trim()
                ));
            }
            "user" => {
                prompt.push_str(&format!(
                    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>",
                    msg.content.trim()
                ));
            }
            "assistant" => {
                prompt.push_str(&format!(
                    "<|start_header_id|>assistant<|end_header_id|>\n\n{}<|eot_id|>",
                    msg.content.trim()
                ));
            }
            _ => {}
        }
    }
    // Signal the model to start generating the assistant reply
    prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    prompt
}
