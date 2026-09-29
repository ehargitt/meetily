use reqwest::{header, Client};
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::path::PathBuf;
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::summary::context_budget::ContextBudget;

const REQUEST_TIMEOUT_DURATION: Duration = Duration::from_secs(300);
/// Slowest prompt-plus-output rate an Ollama request is given time for: roughly an 8B model
/// on CPU or partly offloaded. A full 16k window then gets 300 s + 1024 s.
const OLLAMA_MIN_TOKENS_PER_SEC: u64 = 16;

/// Total time allowed for one request. Ollama gets extra time in proportion to its window,
/// since each chunk now fills that window instead of being truncated to Ollama's default.
fn request_timeout(provider: &LLMProvider, context_budget: Option<ContextBudget>) -> Duration {
    match (provider, context_budget) {
        (LLMProvider::Ollama, Some(budget)) => {
            REQUEST_TIMEOUT_DURATION
                + Duration::from_secs(budget.context_tokens as u64 / OLLAMA_MIN_TOKENS_PER_SEC)
        }
        _ => REQUEST_TIMEOUT_DURATION,
    }
}

async fn await_or_cancel<T>(
    operation: impl Future<Output = T>,
    cancellation_token: Option<&CancellationToken>,
) -> Result<T, String> {
    let Some(token) = cancellation_token else {
        return Ok(operation.await);
    };

    tokio::select! {
        biased;
        _ = token.cancelled() => Err("Summary generation was cancelled".to_string()),
        result = operation => Ok(result),
    }
}

// Generic structure for OpenAI-compatible API chat messages
#[derive(Debug, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

// Generic structure for OpenAI-compatible API chat requests
#[derive(Debug, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
}

/// Build OpenAI-compat JSON body (every provider except Claude, Ollama and BuiltInAI).
pub fn build_openai_compat_chat_body(
    provider: &LLMProvider,
    model_name: &str,
    system_prompt: &str,
    user_prompt: &str,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
) -> serde_json::Value {
    let (max_tokens_val, temperature_val, top_p_val) = if *provider == LLMProvider::CustomOpenAI {
        (max_tokens, temperature, top_p)
    } else {
        (None, None, None)
    };

    serde_json::json!(ChatRequest {
        model: model_name.to_string(),
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: system_prompt.to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: user_prompt.to_string(),
            }
        ],
        max_tokens: max_tokens_val,
        temperature: temperature_val,
        top_p: top_p_val,
    })
}

#[derive(Debug, Serialize)]
struct OllamaChatOptions {
    num_ctx: usize,
    /// Caps the completion at the budget's output reserve so prompt and output always fit
    /// `num_ctx`; unbounded output would make Ollama shift out the start of the prompt.
    num_predict: usize,
}

/// Request body for Ollama's native `/api/chat`. The OpenAI-compatible endpoint cannot set
/// `num_ctx`, so Ollama would run at its small default window and silently drop the start of
/// any longer prompt.
#[derive(Debug, Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
    think: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<OllamaChatOptions>,
}

/// Build the Ollama `/api/chat` body: non-streaming, thinking disabled, and with a budget,
/// `options.num_ctx` = its window and `options.num_predict` = its output reserve.
pub fn build_ollama_chat_body(
    model_name: &str,
    system_prompt: &str,
    user_prompt: &str,
    context_budget: Option<ContextBudget>,
) -> serde_json::Value {
    serde_json::json!(OllamaChatRequest {
        model: model_name.to_string(),
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: system_prompt.to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: user_prompt.to_string(),
            },
        ],
        stream: false,
        think: false,
        options: context_budget.map(|budget| OllamaChatOptions {
            num_ctx: budget.context_tokens,
            num_predict: budget.output_reserve_tokens,
        }),
    })
}

#[derive(Deserialize, Debug)]
struct OllamaChatResponse {
    message: OllamaResponseMessage,
    #[serde(default)]
    done_reason: Option<String>,
    #[serde(default)]
    prompt_eval_count: Option<usize>,
    #[serde(default)]
    eval_count: Option<usize>,
}

#[derive(Deserialize, Debug)]
struct OllamaResponseMessage {
    #[serde(default)]
    content: String,
    #[serde(default)]
    thinking: Option<String>,
}

impl OllamaChatResponse {
    fn completion(&self) -> LlmCompletion {
        LlmCompletion {
            content: self.message.content.trim().to_string(),
            reasoning_stripped: self
                .message
                .thinking
                .as_deref()
                .is_some_and(|thinking| !thinking.trim().is_empty()),
        }
    }

    /// Whether prompt and completion together reached the `num_ctx` window: a sign that Ollama
    /// truncated the prompt or shifted the context, dropping the start of the transcript.
    fn filled_context(&self, num_ctx: usize) -> bool {
        match (self.prompt_eval_count, self.eval_count) {
            (Some(prompt), Some(output)) => prompt + output >= num_ctx,
            _ => false,
        }
    }
}

// Generic structure for OpenAI-compatible API chat responses
#[derive(Deserialize, Debug)]
pub struct ChatResponse {
    pub choices: Vec<Choice>,
}

#[derive(Deserialize, Debug)]
pub struct Choice {
    pub message: MessageContent,
}

#[derive(Deserialize, Debug)]
pub struct MessageContent {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub reasoning: Option<String>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
}

// Claude-specific request structure
#[derive(Debug, Serialize)]
pub struct ClaudeRequest {
    pub model: String,
    pub max_tokens: u32,
    pub system: String,
    pub messages: Vec<ChatMessage>,
}

// Claude-specific response structure
#[derive(Deserialize, Debug)]
pub struct ClaudeChatResponse {
    pub content: Vec<ClaudeChatContent>,
}

#[derive(Deserialize, Debug)]
pub struct ClaudeChatContent {
    #[serde(rename = "type")]
    pub block_type: String,
    pub text: Option<String>,
}

impl ClaudeChatResponse {
    fn completion(&self) -> Option<LlmCompletion> {
        let content = self
            .content
            .iter()
            .find(|block| block.block_type == "text")
            .and_then(|block| block.text.as_deref())?
            .trim()
            .to_string();
        Some(LlmCompletion {
            content,
            reasoning_stripped: self.content.iter().any(|block| {
                matches!(block.block_type.as_str(), "thinking" | "redacted_thinking")
            }),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LlmCompletion {
    pub content: String,
    pub reasoning_stripped: bool,
}

impl ChatResponse {
    fn completion(&self) -> Result<LlmCompletion, String> {
        let message = &self
            .choices
            .first()
            .ok_or("No content in LLM response")?
            .message;
        Ok(LlmCompletion {
            content: message
                .content
                .as_deref()
                .unwrap_or_default()
                .trim()
                .to_string(),
            reasoning_stripped: [message.reasoning.as_deref(), message.reasoning_content.as_deref()]
                .into_iter()
                .flatten()
                .any(|reasoning| !reasoning.trim().is_empty()),
        })
    }
}

/// Whether an Ollama error response rejects the `think` field (an older server or a model that
/// cannot switch thinking off), so the request should be retried once without it.
pub(crate) fn ollama_rejects_think(status: reqwest::StatusCode, body: &str) -> bool {
    if !matches!(status.as_u16(), 400 | 422) {
        return false;
    }

    let Ok(value) = serde_json::from_str::<serde_json::Value>(body) else {
        return false;
    };
    let matches_field = |value: Option<&serde_json::Value>| {
        value
            .and_then(serde_json::Value::as_str)
            .is_some_and(|field| {
                field.eq_ignore_ascii_case("think") || field.eq_ignore_ascii_case("reasoning_effort")
            })
    };
    if matches_field(value.get("param"))
        || matches_field(value.get("field"))
        || matches_field(value.get("error").and_then(|error| error.get("param")))
        || matches_field(value.get("error").and_then(|error| error.get("field")))
    {
        return true;
    }

    let matches_message = [
        value.as_str(),
        value.get("error").and_then(serde_json::Value::as_str),
        value.get("message").and_then(serde_json::Value::as_str),
        value
            .get("error")
            .and_then(|error| error.get("message"))
            .and_then(serde_json::Value::as_str),
    ]
    .into_iter()
    .flatten()
    .any(|message| {
        let message = message.to_ascii_lowercase();
        message.contains("reasoning_effort")
            || (message.contains("think")
                && (message.contains("not supported")
                    || message.contains("does not support")
                    || message.contains("invalid")))
    });
    matches_message
}
/// LLM Provider enumeration for multi-provider support
#[derive(Debug, Clone, PartialEq)]
pub enum LLMProvider {
    OpenAI,
    Claude,
    Groq,
    Ollama,
    OpenRouter,
    BuiltInAI,
    CustomOpenAI,
}

impl LLMProvider {
    /// Parse provider from string (case-insensitive)
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "openai" => Ok(Self::OpenAI),
            "claude" => Ok(Self::Claude),
            "groq" => Ok(Self::Groq),
            "ollama" => Ok(Self::Ollama),
            "openrouter" => Ok(Self::OpenRouter),
            "builtin-ai" | "local-llama" | "localllama" => Ok(Self::BuiltInAI),
            "custom-openai" => Ok(Self::CustomOpenAI),
            _ => Err(format!("Unsupported LLM provider: {}", s)),
        }
    }
}

/// Generates a summary using the specified LLM provider
///
/// # Arguments
/// * `client` - Reqwest HTTP client (reused for performance)
/// * `provider` - The LLM provider to use
/// * `model_name` - The specific model to use (e.g., "gpt-4", "claude-3-opus")
/// * `api_key` - API key for the provider (not needed for Ollama)
/// * `system_prompt` - System instructions for the LLM
/// * `user_prompt` - User query/content to process
/// * `ollama_endpoint` - Optional custom Ollama endpoint (defaults to localhost:11434)
/// * `custom_openai_endpoint` - Optional custom OpenAI-compatible endpoint
/// * `max_tokens` - Optional max tokens (for CustomOpenAI provider)
/// * `temperature` - Optional temperature (for CustomOpenAI provider)
/// * `top_p` - Optional top_p (for CustomOpenAI provider)
/// * `context_budget` - Window and output reserve chunking assumed (Ollama sends them as options)
/// * `app_data_dir` - Optional app data directory (for BuiltInAI provider)
/// * `cancellation_token` - Optional token to cancel the request
///
/// The generated visible content and whether private reasoning was removed.
pub(crate) async fn generate_summary(
    client: &Client,
    provider: &LLMProvider,
    model_name: &str,
    api_key: &str,
    system_prompt: &str,
    user_prompt: &str,
    ollama_endpoint: Option<&str>,
    custom_openai_endpoint: Option<&str>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    context_budget: Option<ContextBudget>,
    app_data_dir: Option<&PathBuf>,
    cancellation_token: Option<&CancellationToken>,
) -> Result<LlmCompletion, String> {
    // Check if cancelled before starting
    if let Some(token) = cancellation_token {
        if token.is_cancelled() {
            return Err("Summary generation was cancelled".to_string());
        }
    }

    // Handle BuiltInAI provider separately (uses local sidecar, no HTTP API)
    if provider == &LLMProvider::BuiltInAI {
        let app_data_dir = app_data_dir
            .ok_or_else(|| "app_data_dir is required for BuiltInAI provider".to_string())?;

        return crate::summary::summary_engine::generate_with_builtin(
            app_data_dir,
            model_name,
            system_prompt,
            user_prompt,
            cancellation_token,
        )
        .await
        .map(|content| LlmCompletion {
            content,
            reasoning_stripped: false,
        })
        .map_err(|e| e.to_string());
    }

    let (api_url, mut headers) = match provider {
        LLMProvider::OpenAI => (
            "https://api.openai.com/v1/chat/completions".to_string(),
            header::HeaderMap::new(),
        ),
        LLMProvider::Groq => (
            "https://api.groq.com/openai/v1/chat/completions".to_string(),
            header::HeaderMap::new(),
        ),
        LLMProvider::OpenRouter => (
            "https://openrouter.ai/api/v1/chat/completions".to_string(),
            header::HeaderMap::new(),
        ),
        LLMProvider::Ollama => {
            let host = ollama_endpoint
                .map(|s| s.to_string())
                .unwrap_or_else(|| "http://localhost:11434".to_string());
            (format!("{}/api/chat", host), header::HeaderMap::new())
        }
        LLMProvider::CustomOpenAI => {
            let endpoint = custom_openai_endpoint
                .ok_or_else(|| "Custom OpenAI endpoint not configured".to_string())?;
            (
                format!("{}/chat/completions", endpoint.trim_end_matches('/')),
                header::HeaderMap::new(),
            )
        }
        LLMProvider::Claude => {
            let mut header_map = header::HeaderMap::new();
            header_map.insert(
                "x-api-key",
                api_key
                    .parse()
                    .map_err(|_| "Invalid API key format".to_string())?,
            );
            header_map.insert(
                "anthropic-version",
                "2023-06-01"
                    .parse()
                    .map_err(|_| "Invalid anthropic version".to_string())?,
            );
            ("https://api.anthropic.com/v1/messages".to_string(), header_map)
        }
        LLMProvider::BuiltInAI => {
            // This case is handled earlier with early returns
            unreachable!("BuiltInAI is handled before this match statement")
        }
    };

    // Add authorization header for non-Claude providers
    if provider != &LLMProvider::Claude {
        headers.insert(
            header::AUTHORIZATION,
            format!("Bearer {}", api_key)
                .parse()
                .map_err(|_| "Invalid authorization header".to_string())?,
        );
    }
    headers.insert(
        header::CONTENT_TYPE,
        "application/json"
            .parse()
            .map_err(|_| "Invalid content type".to_string())?,
    );

    // Build request body based on provider
    let request_body = if provider == &LLMProvider::Ollama {
        build_ollama_chat_body(model_name, system_prompt, user_prompt, context_budget)
    } else if provider != &LLMProvider::Claude {
        build_openai_compat_chat_body(
            provider,
            model_name,
            system_prompt,
            user_prompt,
            max_tokens,
            temperature,
            top_p,
        )
    } else {
        serde_json::json!(ClaudeRequest {
            system: system_prompt.to_string(),
            model: model_name.to_string(),
            // Shared budget: on models with thinking enabled by default this
            // covers thinking tokens as well as the answer.
            max_tokens: 8192,
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: user_prompt.to_string(),
            }]
        })
    };

    info!("🐞 LLM Request to {}: model={}", provider_name(provider), model_name);

    let request_timeout = request_timeout(provider, context_budget);
    // Send request with timeout and cancellation support
    let request_future = client
        .post(api_url.clone())
        .headers(headers.clone())
        .json(&request_body)
        .timeout(request_timeout)
        .send();

    // Use tokio::select to race between cancellation and request completion
    let response = if let Some(token) = cancellation_token {
        tokio::select! {
            result = request_future => {
                result.map_err(|e| {
                    if e.is_timeout() {
                        format!(
                            "LLM request timed out after {} seconds",
                            request_timeout.as_secs()
                        )
                    } else {
                        format!("Failed to send request to LLM: {}", e)
                    }
                })?
            }
            _ = token.cancelled() => {
                return Err("Summary generation was cancelled".to_string());
            }
        }
    } else {
        request_future.await.map_err(|e| {
            if e.is_timeout() {
                format!(
                    "LLM request timed out after {} seconds",
                    request_timeout.as_secs()
                )
            } else {
                format!("Failed to send request to LLM: {}", e)
            }
        })?
    };

    let response = if response.status().is_success() {
        response
    } else {
        let status = response.status();
        let error_body = await_or_cancel(response.text(), cancellation_token)
            .await?
            .unwrap_or_else(|error| format!("Failed to read LLM error response body: {error}"));
        if provider != &LLMProvider::Ollama || !ollama_rejects_think(status, &error_body) {
            return Err(format!(
                "LLM API request failed with status {}: {}",
                status, error_body
            ));
        }

        warn!("Ollama rejected think=false; retrying once without it");
        let mut retry_body = request_body;
        retry_body
            .as_object_mut()
            .ok_or_else(|| "Failed to prepare Ollama compatibility retry".to_string())?
            .remove("think");
        let retry_future = client
            .post(api_url)
            .headers(headers)
            .json(&retry_body)
            .timeout(request_timeout)
            .send();
        await_or_cancel(retry_future, cancellation_token)
            .await?
            .map_err(|e| {
                if e.is_timeout() {
                    format!(
                        "LLM retry request timed out after {} seconds",
                        request_timeout.as_secs()
                    )
                } else {
                    format!("Failed to send retry request to LLM: {}", e)
                }
            })?
    };

    if !response.status().is_success() {
        let status = response.status();
        let error_body = await_or_cancel(response.text(), cancellation_token)
            .await?
            .unwrap_or_else(|error| format!("Failed to read LLM error response body: {error}"));
        return Err(format!(
            "LLM API request failed with status {}: {}",
            status, error_body
        ));
    }

    // Parse response based on provider
    if provider == &LLMProvider::Claude {
        let chat_response = await_or_cancel(
            response.json::<ClaudeChatResponse>(),
            cancellation_token,
        )
        .await?
        .map_err(|e| format!("Failed to parse LLM response: {}", e))?;

        info!("🐞 LLM Response received from Claude");

        let completion = chat_response
            .completion()
            .ok_or("No text content in LLM response")?;
        Ok(completion)
    } else if provider == &LLMProvider::Ollama {
        let chat_response = await_or_cancel(
            response.json::<OllamaChatResponse>(),
            cancellation_token,
        )
        .await?
        .map_err(|e| format!("Failed to parse LLM response: {}", e))?;

        info!(
            prompt_tokens = chat_response.prompt_eval_count,
            output_tokens = chat_response.eval_count,
            done_reason = chat_response.done_reason.as_deref(),
            "🐞 LLM Response received from Ollama"
        );
        if let Some(budget) =
            context_budget.filter(|budget| chat_response.filled_context(budget.context_tokens))
        {
            warn!(
                num_ctx = budget.context_tokens,
                prompt_tokens = chat_response.prompt_eval_count,
                output_tokens = chat_response.eval_count,
                "Ollama request filled its context window; the start of the prompt may have been dropped"
            );
        }
        if chat_response.done_reason.as_deref() == Some("length") {
            warn!(
                output_tokens = chat_response.eval_count,
                "Ollama stopped at the output limit (num_predict); the completion may be cut short"
            );
        }
        Ok(chat_response.completion())
    } else {
        let chat_response = await_or_cancel(
            response.json::<ChatResponse>(),
            cancellation_token,
        )
        .await?
        .map_err(|e| format!("Failed to parse LLM response: {}", e))?;

        info!("🐞 LLM Response received from {}", provider_name(provider));
        chat_response.completion()
    }
}

/// Minimal HTTP helpers for tests that stand in for an LLM server on a local socket.
#[cfg(test)]
pub(crate) mod test_http {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    pub(crate) async fn read_http_request(stream: &mut tokio::net::TcpStream) -> Vec<u8> {
        let mut request = Vec::new();
        let mut buffer = [0_u8; 1024];

        loop {
            let bytes_read = stream.read(&mut buffer).await.unwrap();
            assert_ne!(bytes_read, 0, "connection closed before completing request");
            request.extend_from_slice(&buffer[..bytes_read]);

            let Some(headers_end) = request
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .map(|position| position + 4)
            else {
                continue;
            };
            let content_length = std::str::from_utf8(&request[..headers_end])
                .unwrap()
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().unwrap())
                })
                .unwrap_or(0); // bodiless requests such as GET
            if request.len() >= headers_end + content_length {
                return request;
            }
        }
    }

    pub(crate) fn request_json(request: &[u8]) -> serde_json::Value {
        let headers_end = request
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .map(|position| position + 4)
            .expect("request should contain complete headers");
        serde_json::from_slice(&request[headers_end..]).expect("request body should be valid JSON")
    }

    /// The request line's path, e.g. `/api/chat`.
    pub(crate) fn request_path(request: &[u8]) -> String {
        let request_line = request.split(|byte| *byte == b'\r').next().unwrap_or_default();
        String::from_utf8_lossy(request_line)
            .split(' ')
            .nth(1)
            .unwrap_or_default()
            .to_string()
    }

    /// Writes a complete `200 OK` JSON response and closes the connection.
    pub(crate) async fn write_json_response(stream: &mut tokio::net::TcpStream, body: &[u8]) {
        let headers = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        stream.write_all(headers.as_bytes()).await.unwrap();
        stream.write_all(body).await.unwrap();
        stream.flush().await.unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::test_http::{read_http_request, request_json, request_path, write_json_response};
    use super::*;
    use serde_json::json;
    use std::{cell::Cell, task::Poll};
    use tokio::{
        io::AsyncWriteExt,
        net::TcpListener,
        sync::oneshot,
        time::{timeout, Duration},
    };

    #[test]
    fn openai_compatible_bodies_preserve_custom_sampling_only() {
        for provider in [
            LLMProvider::OpenAI,
            LLMProvider::Groq,
            LLMProvider::OpenRouter,
            LLMProvider::CustomOpenAI,
        ] {
            let body = build_openai_compat_chat_body(&provider, "model", "sys", "user", None, None, None);
            assert!(body.get("reasoning_effort").is_none());
            assert!(body.get("options").is_none());
        }
        let openai = build_openai_compat_chat_body(
            &LLMProvider::OpenAI, "model", "sys", "user", Some(12), Some(0.3), Some(0.8),
        );
        assert!(openai.get("max_tokens").is_none());
        let custom = build_openai_compat_chat_body(
            &LLMProvider::CustomOpenAI, "model", "sys", "user", Some(12), Some(0.3), Some(0.8),
        );
        assert_eq!(custom["max_tokens"], 12);
        assert_eq!(custom["temperature"].as_f64(), Some(0.3_f32 as f64));
        assert_eq!(custom["top_p"].as_f64(), Some(0.8_f32 as f64));
    }

    #[test]
    fn compatible_reasoning_is_separate_from_visible_content() {
        for message in [
            json!({"reasoning_content": "private"}),
            json!({"content": null, "reasoning_content": "private"}),
        ] {
            let response: ChatResponse = serde_json::from_value(json!({
                "choices": [{"message": message}]
            }))
            .unwrap();
            assert_eq!(
                response.completion().unwrap(),
                LlmCompletion {
                    content: String::new(),
                    reasoning_stripped: true,
                }
            );
        }
    }

    #[test]
    fn ollama_body_uses_native_chat_with_num_ctx_and_thinking_off() {
        let body = build_ollama_chat_body("llama3.1:8b", "sys", "user", Some(ContextBudget::for_ollama(16_384)));
        assert_eq!(body["model"], "llama3.1:8b");
        assert_eq!(body["stream"], false);
        assert_eq!(body["think"], false);
        assert_eq!(body["options"]["num_ctx"], 16_384);
        assert_eq!(body["options"]["num_predict"], 4096);
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][1]["content"], "user");
        assert!(body.get("reasoning_effort").is_none());

        assert!(build_ollama_chat_body("m", "s", "u", None).get("options").is_none());
    }

    #[test]
    fn ollama_request_timeout_grows_with_its_context_window() {
        let window = |tokens| Some(ContextBudget::for_ollama(tokens));
        assert_eq!(request_timeout(&LLMProvider::Ollama, window(16_384)), Duration::from_secs(300 + 1024));
        assert_eq!(request_timeout(&LLMProvider::Ollama, window(4096)), Duration::from_secs(300 + 256));
        assert_eq!(request_timeout(&LLMProvider::Ollama, None), REQUEST_TIMEOUT_DURATION);
        assert_eq!(
            request_timeout(&LLMProvider::OpenRouter, Some(ContextBudget::for_hosted_model(200_000))),
            REQUEST_TIMEOUT_DURATION
        );
    }

    #[test]
    fn ollama_response_separates_thinking_and_detects_a_full_context() {
        let response: OllamaChatResponse = serde_json::from_value(json!({
            "message": {"role": "assistant", "content": " Summary ", "thinking": "private"},
            "done_reason": "stop",
            "prompt_eval_count": 12_000,
            "eval_count": 4_384
        }))
        .unwrap();
        assert_eq!(
            response.completion(),
            LlmCompletion {
                content: "Summary".to_string(),
                reasoning_stripped: true,
            }
        );
        assert!(response.filled_context(16_384));
        assert!(!response.filled_context(32_768));
    }

    #[test]
    fn ollama_think_rejection_requires_compatible_error() {
        assert!(ollama_rejects_think(
            reqwest::StatusCode::BAD_REQUEST,
            r#"{"error":"\"gpt-oss\" does not support thinking"}"#,
        ));
        assert!(ollama_rejects_think(
            reqwest::StatusCode::BAD_REQUEST,
            r#"{"error":{"param":"think"}}"#,
        ));
        assert!(ollama_rejects_think(
            reqwest::StatusCode::BAD_REQUEST,
            r#"{"error":{"param":"reasoning_effort"}}"#,
        ));
        assert!(ollama_rejects_think(
            reqwest::StatusCode::UNPROCESSABLE_ENTITY,
            r#"{"message":"Unsupported parameter 'ReAsOnInG_EfFoRt'."}"#,
        ));
        assert!(ollama_rejects_think(
            reqwest::StatusCode::BAD_REQUEST,
            r#"{"error":"think value \"none\" is not supported for this model"}"#,
        ));
        assert!(!ollama_rejects_think(
            reqwest::StatusCode::BAD_REQUEST,
            r#"{"message":"invalid model"}"#,
        ));
        assert!(!ollama_rejects_think(
            reqwest::StatusCode::UNAUTHORIZED,
            r#"{"error":{"param":"reasoning_effort"}}"#,
        ));
        assert!(!ollama_rejects_think(
            reqwest::StatusCode::INTERNAL_SERVER_ERROR,
            r#"{"error":"think value \"none\" is not supported"}"#,
        ));
    }

    #[tokio::test]
    async fn cancellation_prevents_compatibility_io_from_being_polled() {
        let cancellation_token = CancellationToken::new();
        cancellation_token.cancel();
        let polled = Cell::new(false);

        let result = await_or_cancel(
            std::future::poll_fn(|_| {
                polled.set(true);
                Poll::<()>::Pending
            }),
            Some(&cancellation_token),
        )
        .await;

        assert_eq!(result, Err("Summary generation was cancelled".to_string()));
        assert!(!polled.get());
    }

    #[tokio::test]
    async fn cancellation_during_failed_ollama_retry_body_returns_promptly() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (retry_headers_sent, mut retry_headers_ready) = oneshot::channel();
        let (release_retry_body, retry_body_released) = oneshot::channel();

        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let request = read_http_request(&mut stream).await;
            let body = request_json(&request);
            assert_eq!(body["think"], false);

            let rejection_body = br#"{"error":{"param":"think"}}"#;
            let rejection_headers = format!(
                "HTTP/1.1 400 Bad Request\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                rejection_body.len()
            );
            stream.write_all(rejection_headers.as_bytes()).await.unwrap();
            stream.write_all(rejection_body).await.unwrap();
            stream.flush().await.unwrap();
            drop(stream);

            let (mut stream, _) = listener.accept().await.unwrap();
            let request = read_http_request(&mut stream).await;
            let body = request_json(&request);
            assert!(body.get("think").is_none());

            stream
                .write_all(
                    b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 4\r\nConnection: close\r\n\r\n",
                )
                .await
                .unwrap();
            stream.flush().await.unwrap();
            let _ = retry_headers_sent.send(());
            let _ = retry_body_released.await;
            let _ = stream.write_all(b"fail").await;
        });

        let client = Client::new();
        let cancellation_token = CancellationToken::new();
        let endpoint = format!("http://{address}");
        let generation = generate_summary(
            &client,
            &LLMProvider::Ollama,
            "model",
            "",
            "system",
            "user",
            Some(&endpoint),
            None,
            None,
            None,
            None,
            None,
            None,
            Some(&cancellation_token),
        );
        tokio::pin!(generation);

        tokio::select! {
            biased;
            result = &mut generation => panic!("generation ended before retry headers: {result:?}"),
            _ = &mut retry_headers_ready => {}
        }

        cancellation_token.cancel();
        let completion = timeout(Duration::from_secs(1), &mut generation).await;
        let _ = release_retry_body.send(());
        let released_completion = if completion.is_err() {
            Some(timeout(Duration::from_secs(1), &mut generation).await)
        } else {
            None
        };
        server.await.unwrap();

        let result = completion.unwrap_or_else(|_| {
            released_completion
                .expect("generation should be awaited after releasing the retry body")
                .expect("generation should finish after releasing the retry body")
        });
        assert_eq!(result, Err("Summary generation was cancelled".to_string()));
    }

    #[tokio::test]
    async fn cancellation_during_successful_ollama_retry_body_returns_promptly() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (retry_headers_sent, mut retry_headers_ready) = oneshot::channel();
        let (release_retry_body, retry_body_released) = oneshot::channel();

        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let request = read_http_request(&mut stream).await;
            let body = request_json(&request);
            assert_eq!(body["think"], false);

            let rejection_body = br#"{"error":{"param":"think"}}"#;
            let rejection_headers = format!(
                "HTTP/1.1 400 Bad Request\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                rejection_body.len()
            );
            stream.write_all(rejection_headers.as_bytes()).await.unwrap();
            stream.write_all(rejection_body).await.unwrap();
            stream.flush().await.unwrap();
            drop(stream);

            let (mut stream, _) = listener.accept().await.unwrap();
            let request = read_http_request(&mut stream).await;
            let body = request_json(&request);
            assert!(body.get("think").is_none());

            let completion_body =
                br#"{"message":{"role":"assistant","content":"Meeting summary."},"done":true}"#;
            let completion_headers = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                completion_body.len()
            );
            stream.write_all(completion_headers.as_bytes()).await.unwrap();
            stream.flush().await.unwrap();
            tokio::time::sleep(Duration::from_millis(100)).await;
            let _ = retry_headers_sent.send(());
            let _ = retry_body_released.await;
            let _ = stream.write_all(completion_body).await;
        });

        let client = Client::new();
        let cancellation_token = CancellationToken::new();
        let endpoint = format!("http://{address}");
        let generation = generate_summary(
            &client,
            &LLMProvider::Ollama,
            "model",
            "",
            "system",
            "user",
            Some(&endpoint),
            None,
            None,
            None,
            None,
            None,
            None,
            Some(&cancellation_token),
        );
        tokio::pin!(generation);

        tokio::select! {
            biased;
            result = &mut generation => panic!("generation ended before retry headers: {result:?}"),
            _ = &mut retry_headers_ready => {}
        }

        cancellation_token.cancel();
        let completion = timeout(Duration::from_secs(1), &mut generation).await;
        let _ = release_retry_body.send(());
        if completion.is_err() {
            let _ = timeout(Duration::from_secs(1), &mut generation).await;
        }
        server.await.unwrap();

        let result =
            completion.expect("generation should observe cancellation before retry body release");
        assert_eq!(result, Err("Summary generation was cancelled".to_string()));
    }

    #[tokio::test]
    async fn ollama_retry_without_think_keeps_num_ctx_and_returns_completion() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let request = read_http_request(&mut stream).await;
            let body = request_json(&request);
            assert_eq!(request_path(&request), "/api/chat");
            assert_eq!(body["think"], false);
            assert_eq!(body["options"]["num_ctx"], 8192);

            let rejection_body = br#"{"error":{"param":"think"}}"#;
            let rejection_headers = format!(
                "HTTP/1.1 400 Bad Request\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                rejection_body.len()
            );
            stream.write_all(rejection_headers.as_bytes()).await.unwrap();
            stream.write_all(rejection_body).await.unwrap();
            stream.flush().await.unwrap();
            drop(stream);

            let (mut stream, _) = listener.accept().await.unwrap();
            let request = read_http_request(&mut stream).await;
            let body = request_json(&request);
            assert_eq!(request_path(&request), "/api/chat");
            assert!(body.get("think").is_none());
            assert_eq!(body["options"]["num_ctx"], 8192, "the retry keeps the context window");
            assert_eq!(body["options"]["num_predict"], 2048);

            write_json_response(
                &mut stream,
                br#"{"message":{"role":"assistant","content":"Meeting summary."},"done":true}"#,
            )
            .await;
        });

        let client = Client::new();
        let cancellation_token = CancellationToken::new();
        let endpoint = format!("http://{address}");
        let completion = timeout(
            Duration::from_secs(1),
            generate_summary(
                &client,
                &LLMProvider::Ollama,
                "model",
                "",
                "system",
                "user",
                Some(&endpoint),
                None,
                None,
                None,
                None,
                Some(ContextBudget::for_ollama(8192)),
                None,
                Some(&cancellation_token),
            ),
        )
        .await;
        server.await.unwrap();

        assert_eq!(
            completion
                .expect("generation should finish after the compatibility retry")
                .unwrap(),
            LlmCompletion {
                content: "Meeting summary.".to_string(),
                reasoning_stripped: false,
            }
        );
    }

    #[test]
    fn claude_response_uses_exact_block_types() {
        let response: ClaudeChatResponse = serde_json::from_value(json!({
            "content": [
                {"type": "analysis", "text": "not visible"},
                {"type": "thinking", "thinking": "private"},
                {"type": "text", "text": "Meeting summary."}
            ]
        }))
        .unwrap();
        assert_eq!(
            response.completion(),
            Some(LlmCompletion {
                content: "Meeting summary.".to_string(),
                reasoning_stripped: true,
            })
        );
    }
}

/// Helper function to get provider name for logging
fn provider_name(provider: &LLMProvider) -> &str {
    match provider {
        LLMProvider::OpenAI => "OpenAI",
        LLMProvider::Claude => "Claude",
        LLMProvider::Groq => "Groq",
        LLMProvider::Ollama => "Ollama",
        LLMProvider::BuiltInAI => "Built-in AI",
        LLMProvider::OpenRouter => "OpenRouter",
        LLMProvider::CustomOpenAI => "Custom OpenAI",
    }
}


