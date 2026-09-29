use crate::database::repositories::{
    meeting::MeetingsRepository, setting::SettingsRepository, summary::SummaryProcessesRepository,
};
use crate::summary::context_budget::{ContextBudget, OLLAMA_FALLBACK_CONTEXT};
use crate::summary::llm_client::LLMProvider;
use crate::summary::language_detection::detect_summary_language;
use crate::summary::metadata::read_detected_summary_language_from_metadata;
use crate::summary::processor::{
    clean_llm_markdown_detailed, extract_meeting_name_from_markdown, generate_meeting_summary,
    CachedEnglishSummary,
    language_name_from_code, require_visible_markdown,
};
use crate::summary::summary_engine::models;
use crate::summary::templates::{self, Template};
use crate::ollama::metadata::ModelMetadataCache;
use crate::openrouter;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Manager};
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};

static METADATA_CACHE: LazyLock<ModelMetadataCache> =
    LazyLock::new(|| ModelMetadataCache::new(Duration::from_secs(300)));

#[derive(Clone)]
struct RegisteredCancellation {
    started_at: DateTime<Utc>,
    token: CancellationToken,
}

static CANCELLATION_REGISTRY: LazyLock<Arc<Mutex<HashMap<String, RegisteredCancellation>>>> =
    LazyLock::new(|| Arc::new(Mutex::new(HashMap::new())));

/// Strips the first `#` heading line; returns "" if no `#` is found.
fn strip_leading_title(markdown: &str) -> String {
    if let Some(hash_pos) = markdown.find('#') {
        let body_start = markdown[hash_pos..]
            .find('\n')
            .map_or(markdown.len(), |line_end| hash_pos + line_end);
        markdown[body_start..].trim_start().to_string()
    } else {
        String::new()
    }
}

/// Strips the leading H1 (`# Title\n...`) only when the markdown starts with one.
/// No-op on already-stripped values, values starting with `## Subheading`, or values
/// without any heading. Avoids the silent-empty-return case where `strip_leading_title`
/// returns "" for input lacking a leading `#`.
fn strip_title_if_present(markdown: &str) -> String {
    if markdown.trim_start().starts_with("# ") {
        strip_leading_title(markdown)
    } else {
        markdown.to_string()
    }
}

const ENGLISH_CACHE_FIELD: &str = "english_cache";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct SummaryCacheSource {
    transcript_fingerprint: String,
    custom_prompt_fingerprint: String,
    template_id: String,
    template_fingerprint: String,
    context_budget: Option<ContextBudget>,
    model_provider: String,
    model_name: String,
    ollama_endpoint: Option<String>,
    custom_openai_endpoint: Option<String>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct EnglishSummaryCache {
    markdown: String,
    source: SummaryCacheSource,
    output_language: Option<String>,
    /// The English markdown was built from a cut-off combine (`combine_truncated`); a rerun
    /// that reuses it must report that too. Rows written before this field read as false.
    #[serde(default)]
    combine_truncated: bool,
}

fn stable_text_fingerprint(text: &str) -> String {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut hash = FNV_OFFSET;
    for byte in text.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    format!("{:016x}:{}", hash, text.len())
}

#[allow(clippy::too_many_arguments)]
fn build_summary_cache_source(
    text: &str,
    custom_prompt: &str,
    template_id: &str,
    template_fingerprint: &str,
    context_budget: Option<ContextBudget>,
    model_provider: &str,
    model_name: &str,
    ollama_endpoint: Option<&str>,
    custom_openai_endpoint: Option<&str>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
) -> SummaryCacheSource {
    SummaryCacheSource {
        transcript_fingerprint: stable_text_fingerprint(text),
        custom_prompt_fingerprint: stable_text_fingerprint(custom_prompt),
        template_id: template_id.to_string(),
        template_fingerprint: template_fingerprint.to_string(),
        context_budget,
        model_provider: model_provider.to_string(),
        model_name: model_name.to_string(),
        ollama_endpoint: ollama_endpoint.map(str::to_string),
        custom_openai_endpoint: custom_openai_endpoint.map(str::to_string),
        max_tokens,
        temperature,
        top_p,
    }
}

fn template_cache_fingerprint(template: &Template) -> String {
    let rendered_template = format!(
        "{}\n---SECTION-INSTRUCTIONS---\n{}",
        template.to_markdown_structure(),
        template.to_section_instructions()
    );
    stable_text_fingerprint(&rendered_template)
}

fn normalise_summary_language_for_cache(summary_language: Option<&str>) -> Option<String> {
    language_name_from_code(summary_language?.trim()).map(str::to_string)
}

fn build_summary_result_json(
    final_markdown: &str,
    english_markdown: &str,
    source: SummaryCacheSource,
    output_language: Option<&str>,
    reasoning_stripped: bool,
    normalization_fallback: bool,
    combine_truncated: bool,
) -> Result<serde_json::Value, String> {
    let cleaned_final = clean_llm_markdown_detailed(final_markdown);
    require_visible_markdown("Final summary", &cleaned_final)?;
    let markdown = strip_title_if_present(&cleaned_final.markdown);
    if markdown.trim().is_empty() {
        return Err("Final summary contains no visible content after title removal".to_string());
    }

    let cleaned_english = clean_llm_markdown_detailed(english_markdown);
    require_visible_markdown("English summary", &cleaned_english)?;

    Ok(serde_json::json!({
        "markdown": markdown,
        ENGLISH_CACHE_FIELD: EnglishSummaryCache {
            markdown: cleaned_english.markdown,
            source,
            output_language: normalise_summary_language_for_cache(output_language),
            combine_truncated,
        },
        "reasoning_stripped": reasoning_stripped
            || cleaned_final.reasoning_stripped
            || cleaned_english.reasoning_stripped,
        "normalization_fallback": normalization_fallback,
        "combine_truncated": combine_truncated,
    }))
}

/// Parses a `summary_processes.result` JSON blob and extracts a cached English
/// summary only when it was produced from exactly the same source inputs and
/// the user is switching to a different non-English target language.
fn extract_cached_english_markdown(
    raw: &str,
    expected_source: &SummaryCacheSource,
    requested_language: Option<&str>,
) -> Result<Option<CachedEnglishSummary>, serde_json::Error> {
    let requested_language = match normalise_summary_language_for_cache(requested_language) {
        Some(language) if language != "English" => language,
        _ => return Ok(None),
    };

    let value: serde_json::Value = serde_json::from_str(raw)?;
    let Some(cache_value) = value.get(ENGLISH_CACHE_FIELD) else {
        return Ok(None);
    };

    let cache: EnglishSummaryCache = match serde_json::from_value(cache_value.clone()) {
        Ok(cache) => cache,
        Err(_) => return Ok(None),
    };

    if cache.source != *expected_source {
        return Ok(None);
    }

    if cache.output_language.as_deref() == Some(requested_language.as_str()) {
        return Ok(None);
    }

    let markdown = cache.markdown.trim();
    if markdown.is_empty() {
        Ok(None)
    } else {
        Ok(Some(CachedEnglishSummary {
            markdown: cache.markdown,
            combine_truncated: cache.combine_truncated,
        }))
    }
}

/// Summary service - handles all summary generation logic
pub struct SummaryService;

impl SummaryService {
    /// Registers a new cancellation token for a meeting.
    pub(crate) fn register_cancellation_token(
        meeting_id: &str,
        started_at: DateTime<Utc>,
    ) -> CancellationToken {
        let token = CancellationToken::new();
        let mut registry = CANCELLATION_REGISTRY
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(previous) = registry.insert(
            meeting_id.to_string(),
            RegisteredCancellation {
                started_at,
                token: token.clone(),
            },
        ) {
            previous.token.cancel();
        }
        info!("Registered cancellation token for meeting: {}", meeting_id);
        token
    }

    /// Cancels only the active generation identified by `started_at`.
    pub fn cancel_summary(meeting_id: &str, started_at: DateTime<Utc>) -> bool {
        let registry = CANCELLATION_REGISTRY
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(entry) = registry.get(meeting_id).filter(|entry| entry.started_at == started_at)
        {
            info!("Cancelling summary generation for meeting: {}", meeting_id);
            entry.token.cancel();
            true
        } else {
            warn!("No active summary generation found for meeting: {}", meeting_id);
            false
        }
    }

    /// Cleans up only the matching generation token after processing completes.
    fn cleanup_cancellation_token(meeting_id: &str, started_at: DateTime<Utc>) {
        let mut registry = CANCELLATION_REGISTRY
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if registry
            .get(meeting_id)
            .is_some_and(|entry| entry.started_at == started_at)
        {
            registry.remove(meeting_id);
            info!("Cleaned up cancellation token for meeting: {}", meeting_id);
        }
    }

    async fn read_detected_summary_language(
        pool: &SqlitePool,
        meeting_id: &str,
    ) -> Option<String> {
        let meeting = match MeetingsRepository::get_meeting_metadata(pool, meeting_id).await {
            Ok(Some(meeting)) => meeting,
            Ok(None) => {
                warn!("Meeting not found while reading detected summary language: {}", meeting_id);
                return None;
            }
            Err(e) => {
                warn!(
                    "Failed to read meeting metadata for detected summary language (meeting_id={}): {}",
                    meeting_id, e
                );
                return None;
            }
        };

        let Some(folder_path) = meeting.folder_path.filter(|p| !p.trim().is_empty()) else {
            return None;
        };

        match read_detected_summary_language_from_metadata(Path::new(&folder_path)) {
            Ok(language) => language,
            Err(e) => {
                warn!(
                    "Failed to read detected summary language metadata for meeting_id={}: {}",
                    meeting_id, e
                );
                None
            }
        }
    }

    fn detect_summary_language_from_text(text: &str) -> Option<String> {
        let transcript_texts = [text.to_string()];
        let detection = detect_summary_language(&transcript_texts);
        match &detection.language {
            Some(language) => {
                info!("Detected transcript summary language for normalization: {}", language);
            }
            None => {
                info!(
                    "Transcript summary language unknown for normalization: {:?}",
                    detection.reason
                );
            }
        }
        detection.language
    }

    /// Processes transcript in the background and generates summary
    ///
    /// This function is designed to be spawned as an async task and does not block
    /// the main thread. It updates the database with progress and results.
    ///
    /// # Arguments
    /// * `_app` - Tauri app handle (for future use)
    /// * `pool` - SQLx connection pool
    /// * `meeting_id` - Unique identifier for the meeting
    /// * `text` - Full transcript text
    /// * `model_provider` - LLM provider name (e.g., "ollama", "openai")
    /// * `model_name` - Specific model (e.g., "gpt-4", "llama3.2:latest")
    /// * `custom_prompt` - Optional user-provided context
    /// * `template_id` - Template identifier (e.g., "daily_standup", "standard_meeting")
    pub async fn process_transcript_background<R: tauri::Runtime>(
        _app: AppHandle<R>,
        pool: SqlitePool,
        meeting_id: String,
        started_at: DateTime<Utc>,
        cancellation_token: CancellationToken,
        text: String,
        model_provider: String,
        model_name: String,
        custom_prompt: String,
        template_id: String,
        summary_language: Option<String>,
    ) {
        let start_time = Instant::now();
        info!(
            "Starting background processing for meeting_id: {}",
            meeting_id
        );

        // Parse provider
        let provider = match LLMProvider::from_str(&model_provider) {
            Ok(p) => p,
            Err(e) => {
                Self::fail_and_cleanup(&pool, &meeting_id, started_at, &e).await;
                return;
            }
        };

        // Validate and setup api_key, Flexible for Ollama, BuiltInAI, and CustomOpenAI
        let api_key = if provider == LLMProvider::Ollama || provider == LLMProvider::BuiltInAI || provider == LLMProvider::CustomOpenAI {
            // These providers don't require API keys from the standard database column
            String::new()
        } else {
            match SettingsRepository::get_api_key(&pool, &model_provider).await {
                Ok(Some(key)) if !key.is_empty() => key,
                Ok(None) | Ok(Some(_)) => {
                    let err_msg = format!("API key not found for {}", &model_provider);
                    Self::fail_and_cleanup(&pool, &meeting_id, started_at, &err_msg).await;
                    return;
                }
                Err(e) => {
                    let err_msg = format!("Failed to retrieve API key for {}: {}", &model_provider, e);
                    Self::fail_and_cleanup(&pool, &meeting_id, started_at, &err_msg).await;
                    return;
                }
            }
        };

        // Get Ollama endpoint if provider is Ollama
        let ollama_endpoint = if provider == LLMProvider::Ollama {
            match SettingsRepository::get_model_config(&pool).await {
                Ok(Some(config)) => config.ollama_endpoint,
                Ok(None) => None,
                Err(e) => {
                    info!("Failed to retrieve Ollama endpoint: {}, using default", e);
                    None
                }
            }
        } else {
            None
        };

        // Get CustomOpenAI config if provider is CustomOpenAI
        let (custom_openai_endpoint, custom_openai_api_key, custom_openai_max_tokens, custom_openai_temperature, custom_openai_top_p) =
            if provider == LLMProvider::CustomOpenAI {
                match SettingsRepository::get_custom_openai_config(&pool).await {
                    Ok(Some(config)) => {
                        info!("✓ Using custom OpenAI endpoint: {}", config.endpoint);
                        (
                            Some(config.endpoint),
                            config.api_key,
                            config.max_tokens.map(|t| t as u32),
                            config.temperature,
                            config.top_p,
                        )
                    }
                    Ok(None) => {
                        let err_msg = "Custom OpenAI provider selected but no configuration found";
                        Self::fail_and_cleanup(&pool, &meeting_id, started_at, err_msg).await;
                        return;
                    }
                    Err(e) => {
                        let err_msg = format!("Failed to retrieve custom OpenAI config: {}", e);
                        Self::fail_and_cleanup(&pool, &meeting_id, started_at, &err_msg).await;
                        return;
                    }
                }
            } else {
                (None, None, None, None, None)
            };

        // For CustomOpenAI, use its API key (if any) instead of the empty string
        let final_api_key = if provider == LLMProvider::CustomOpenAI {
            custom_openai_api_key.unwrap_or_default()
        } else {
            api_key
        };

        // The context window the model runs with. Chunking and Ollama's num_ctx both come from it.
        let context_budget = match Self::select_context_budget(
            &provider,
            &model_name,
            ollama_endpoint.as_deref(),
            openrouter::MODELS_URL,
        )
        .await
        {
            Ok(budget) => budget,
            Err(err_msg) => {
                Self::fail_and_cleanup(&pool, &meeting_id, started_at, &err_msg).await;
                return;
            }
        };
        if let Some(budget) = context_budget {
            info!(
                model = %model_name,
                context_tokens = budget.context_tokens,
                output_reserve_tokens = budget.output_reserve_tokens,
                "Using summary context budget"
            );
        }

        // Get app data directory for BuiltInAI provider
        let app_data_dir = _app.path().app_data_dir().ok();

        if let Some(code) = &summary_language {
            info!("📝 Summary language preference: {}", code);
        }

        let detected_summary_language =
            Self::read_detected_summary_language(&pool, &meeting_id)
                .await
                .or_else(|| Self::detect_summary_language_from_text(&text));

        if let Some(code) = &detected_summary_language {
            info!("📝 Detected transcript summary language: {}", code);
        }

        let template = match templates::get_template(&template_id) {
            Ok(template) => template,
            Err(e) => {
                let err_msg = format!("Failed to load template '{}': {}", template_id, e);
                Self::fail_and_cleanup(&pool, &meeting_id, started_at, &err_msg).await;
                return;
            }
        };
        let template_fingerprint = template_cache_fingerprint(&template);

        let cache_source = build_summary_cache_source(
            &text,
            &custom_prompt,
            &template_id,
            &template_fingerprint,
            context_budget,
            &model_provider,
            &model_name,
            ollama_endpoint.as_deref(),
            custom_openai_endpoint.as_deref(),
            custom_openai_max_tokens,
            custom_openai_temperature,
            custom_openai_top_p,
        );

        let cached_english = match SummaryProcessesRepository::get_summary_data(&pool, &meeting_id).await {
            Err(e) => {
                warn!(
                    "Failed to load prior summary row for cache lookup (meeting_id={}): {}. Falling back to full pass-1 generation.",
                    meeting_id, e
                );
                None
            }
            Ok(None) => None,
            Ok(Some(process)) => process.result.and_then(|raw| {
                match extract_cached_english_markdown(
                    &raw,
                    &cache_source,
                    summary_language.as_deref(),
                ) {
                    Ok(opt) => opt,
                    Err(e) => {
                        warn!(
                            "Cached summary result for meeting_id={} is not valid JSON ({}); ignoring cache.",
                            meeting_id, e
                        );
                        None
                    }
                }
            }),
        };

        let client = reqwest::Client::new();
        let result = generate_meeting_summary(
            &client,
            &provider,
            &model_name,
            &final_api_key,
            &text,
            &custom_prompt,
            &template_id,
            &template,
            context_budget,
            ollama_endpoint.as_deref(),
            custom_openai_endpoint.as_deref(),
            custom_openai_max_tokens,
            custom_openai_temperature,
            custom_openai_top_p,
            app_data_dir.as_ref(),
            Some(&cancellation_token),
            summary_language.as_deref(),
            detected_summary_language.as_deref(),
            cached_english.as_ref(),
        )
        .await;

        let duration = start_time.elapsed().as_secs_f64();

        match result {
            Ok(generated) => {
                info!(
                    "✓ Successfully processed {} chunks for meeting_id: {}. Duration: {:.2}s",
                    generated.successful_chunk_count, meeting_id, duration
                );
                let result_json = match build_summary_result_json(
                    &generated.final_markdown,
                    &generated.english_markdown,
                    cache_source,
                    summary_language.as_deref(),
                    generated.reasoning_stripped,
                    generated.normalization_fallback,
                    generated.combine_truncated,
                ) {
                    Ok(result) => result,
                    Err(error) => {
                        Self::update_process_failed(&pool, &meeting_id, started_at, &error).await;
                        Self::cleanup_cancellation_token(&meeting_id, started_at);
                        return;
                    }
                };

                match SummaryProcessesRepository::update_process_completed(
                    &pool,
                    &meeting_id,
                    started_at,
                    result_json,
                    generated.successful_chunk_count,
                    duration,
                )
                .await
                {
                    Ok(true) => {
                        if let Some(name) =
                            extract_meeting_name_from_markdown(&generated.final_markdown)
                                .filter(|name| !name.is_empty())
                        {
                            if let Err(error) =
                                MeetingsRepository::update_meeting_name(&pool, &meeting_id, &name)
                                    .await
                            {
                                error!("Failed to update meeting name for {}: {}", meeting_id, error);
                            }
                        }
                        info!("Summary saved successfully for meeting_id: {}", meeting_id);
                    }
                    Ok(false) => warn!("Skipped stale summary completion for meeting_id: {}", meeting_id),
                    Err(error) => error!(
                        "Failed to save completed process for {}: {}",
                        meeting_id, error
                    ),
                }
            }
            Err(error) if cancellation_token.is_cancelled() => {
                match SummaryProcessesRepository::update_process_cancelled(
                    &pool,
                    &meeting_id,
                    started_at,
                )
                .await
                {
                    Ok(false) => warn!("Skipped stale summary cancellation for meeting_id: {}", meeting_id),
                    Ok(true) => info!("Summary generation was cancelled for meeting_id: {}", meeting_id),
                    Err(db_error) => error!(
                        "Failed to update DB status to cancelled for {}: {}",
                        meeting_id, db_error
                    ),
                }
            }
            Err(error) => {
                Self::update_process_failed(&pool, &meeting_id, started_at, &error).await;
            }
        }
        Self::cleanup_cancellation_token(&meeting_id, started_at);
    }

    /// The context budget chunking should respect for `provider`/`model_name`, or `None` to
    /// send the whole transcript in one request. `Err` only for an unknown built-in model.
    ///
    /// OpenRouter windows come from its catalogue at `openrouter_models_url`; when that cannot
    /// be read the summary runs single-pass as before. OpenAI, Claude, Groq and custom
    /// OpenAI-compatible servers stay single-pass: nothing in the app reports or configures
    /// their context size (`CustomOpenAIConfig` has no such field).
    async fn select_context_budget(
        provider: &LLMProvider,
        model_name: &str,
        ollama_endpoint: Option<&str>,
        openrouter_models_url: &str,
    ) -> Result<Option<ContextBudget>, String> {
        match provider {
            LLMProvider::Ollama => {
                let model_max_context =
                    match METADATA_CACHE.get_or_fetch(model_name, ollama_endpoint).await {
                        Ok(metadata) => metadata.context_size,
                        Err(e) => {
                            warn!(
                                "Failed to fetch context for {}: {}. Using {} tokens",
                                model_name, e, OLLAMA_FALLBACK_CONTEXT
                            );
                            OLLAMA_FALLBACK_CONTEXT
                        }
                    };
                Ok(Some(ContextBudget::for_ollama(model_max_context)))
            }
            LLMProvider::BuiltInAI => models::get_model_by_name(model_name)
                .map(|model_def| {
                    Some(ContextBudget::for_builtin(
                        model_def.context_size,
                        models::DEFAULT_MAX_TOKENS,
                    ))
                })
                .ok_or_else(|| format!("Unknown built-in model: {}", model_name)),
            LLMProvider::OpenRouter => {
                let client = reqwest::Client::new();
                match openrouter::fetch_context_length(&client, openrouter_models_url, model_name)
                    .await
                {
                    Ok(Some(context_length)) => {
                        Ok(Some(ContextBudget::for_hosted_model(context_length as usize)))
                    }
                    Ok(None) => {
                        warn!(
                            "OpenRouter lists no context length for {}; summarizing in one request",
                            model_name
                        );
                        Ok(None)
                    }
                    Err(e) => {
                        warn!("{}; summarizing {} in one request", e, model_name);
                        Ok(None)
                    }
                }
            }
            LLMProvider::OpenAI
            | LLMProvider::Claude
            | LLMProvider::Groq
            | LLMProvider::CustomOpenAI => Ok(None),
        }
    }

    /// Updates the summary process status to failed with error message
    async fn fail_and_cleanup(
        pool: &SqlitePool,
        meeting_id: &str,
        started_at: DateTime<Utc>,
        error_msg: &str,
    ) {
        Self::update_process_failed(pool, meeting_id, started_at, error_msg).await;
        Self::cleanup_cancellation_token(meeting_id, started_at);
    }

    async fn update_process_failed(
        pool: &SqlitePool,
        meeting_id: &str,
        started_at: DateTime<Utc>,
        error_msg: &str,
    ) {
        error!(
            "Processing failed for meeting_id {}: {}",
            meeting_id, error_msg
        );
        match SummaryProcessesRepository::update_process_failed(
            pool,
            meeting_id,
            started_at,
            error_msg,
        )
        .await
        {
            Ok(false) => warn!("Skipped stale summary failure for meeting_id: {}", meeting_id),
            Ok(true) => {}
            Err(e) => error!(
                "Failed to update DB status to failed for {}: {}",
                meeting_id, e
            ),
        }
    }


}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stale_cleanup_keeps_the_replacement_cancellation_token() {
        let meeting_id = format!("summary-test-{}", uuid::Uuid::new_v4());
        let first_started_at = Utc::now();
        let second_started_at = first_started_at + chrono::Duration::nanoseconds(1);
        let first = SummaryService::register_cancellation_token(&meeting_id, first_started_at);
        let second = SummaryService::register_cancellation_token(&meeting_id, second_started_at);

        assert!(first.is_cancelled());
        SummaryService::cleanup_cancellation_token(&meeting_id, first_started_at);
        assert!(SummaryService::cancel_summary(&meeting_id, second_started_at));
        assert!(second.is_cancelled());
        SummaryService::cleanup_cancellation_token(&meeting_id, second_started_at);
    }

    #[test]
    fn test_strip_leading_title_with_body() {
        let input = "# Meeting Title\nThis is the body.\nMore content.";
        let result = strip_leading_title(input);
        assert_eq!(result, "This is the body.\nMore content.");
    }

    #[test]
    fn test_strip_leading_title_only() {
        let input = "# Meeting Title";
        let result = strip_leading_title(input);
        assert_eq!(result, "");
    }

    #[test]
    fn test_strip_leading_title_no_heading() {
        let input = "No heading here.\nJust body.";
        let result = strip_leading_title(input);
        assert_eq!(result, "");
    }

    #[test]
    fn test_strip_leading_title_multiline_body() {
        let input = "# Title\n## Subheading\nParagraph 1\n\nParagraph 2";
        let result = strip_leading_title(input);
        assert_eq!(result, "## Subheading\nParagraph 1\n\nParagraph 2");
    }

    #[test]
    fn test_strip_leading_title_empty_after_heading() {
        let input = "# Title\n";
        let result = strip_leading_title(input);
        assert_eq!(result, "");
    }

    #[test]
    fn test_strip_leading_title_whitespace_after_heading() {
        let input = "# Title\n   \n Body with leading spaces";
        let result = strip_leading_title(input);
        assert_eq!(result, "Body with leading spaces");
    }

    #[test]
    fn test_strip_title_if_present_preserves_already_stripped() {
        assert_eq!(strip_title_if_present("## Action Items\nfoo"), "## Action Items\nfoo");
    }

    #[test]
    fn test_strip_title_if_present_strips_leading_h1() {
        assert_eq!(strip_title_if_present("# Meeting Title\n## Action Items\nfoo"), "## Action Items\nfoo");
    }

    #[test]
    fn test_strip_title_if_present_no_heading_preserved() {
        // Distinct from strip_leading_title which returns "" — this preserves input.
        assert_eq!(strip_title_if_present("Just body text"), "Just body text");
    }

    #[test]
    fn test_strip_title_if_present_hash_no_space_preserved() {
        // `#NoSpace` is not a markdown H1 — preserve.
        assert_eq!(strip_title_if_present("#NoSpace\nbody"), "#NoSpace\nbody");
    }

    #[test]
    fn test_strip_title_if_present_mid_document_h1_preserved() {
        // H1 after body content must NOT be stripped — guards the asymmetry where
        // extract_meeting_name_from_markdown scans every line for "# ".
        let input = "Some paragraph\n\n# H1 on line 3\n## Section\nbody";
        assert_eq!(strip_title_if_present(input), input);
    }

    #[test]
    fn test_strip_title_if_present_leading_whitespace_h1_stripped() {
        assert_eq!(
            strip_title_if_present("  # Title\n## Section\nbody"),
            "## Section\nbody"
        );
    }

    fn sample_cache_source() -> SummaryCacheSource {
        let template_fingerprint = stable_text_fingerprint("standard template prompt");
        build_summary_cache_source(
            "transcript body",
            "custom prompt",
            "standard_meeting",
            &template_fingerprint,
            Some(ContextBudget::for_ollama(8192)),
            "ollama",
            "gemma3:1b",
            Some("http://localhost:11434"),
            None,
            None,
            None,
            None,
        )
    }

    fn test_template(section_title: &str) -> Template {
        Template {
            name: "Test".to_string(),
            description: "Test template".to_string(),
            sections: vec![crate::summary::templates::TemplateSection {
                title: section_title.to_string(),
                instruction: "Summarize this section".to_string(),
                format: "paragraph".to_string(),
                item_format: None,
                example_item_format: None,
            }],
        }
    }

    #[test]
    fn test_template_cache_fingerprint_changes_with_rendered_template() {
        assert_ne!(
            template_cache_fingerprint(&test_template("Summary")),
            template_cache_fingerprint(&test_template("Decisions"))
        );
    }

    #[test]
    fn test_legacy_english_markdown_field_is_cache_miss() {
        let raw = serde_json::json!({
            "markdown": "translated",
            "english_markdown": "# Old English\nBody"
        })
        .to_string();

        assert_eq!(
            extract_cached_english_markdown(&raw, &sample_cache_source(), Some("de")).unwrap().map(|cache| cache.markdown),
            None
        );
    }

    #[test]
    fn test_matching_source_changed_translation_target_reuses_cache() {
        let source = sample_cache_source();
        let raw = build_summary_result_json(
            "# Reunion\n## Points\nBonjour",
            "# Meeting\n## Points\nHello",
            source.clone(),
            Some("fr"),
            false,
            false,
            false,
        )
        .unwrap()
        .to_string();

        assert_eq!(
            extract_cached_english_markdown(&raw, &source, Some("de")).unwrap().map(|cache| cache.markdown),
            Some("# Meeting\n## Points\nHello".to_string())
        );
    }

    #[test]
    fn test_same_language_regeneration_rejects_cache() {
        let source = sample_cache_source();
        let raw = build_summary_result_json(
            "# Reunion\n## Points\nBonjour",
            "# Meeting\n## Points\nHello",
            source.clone(),
            Some("fr"),
            false,
            false,
            false,
        )
        .unwrap()
        .to_string();

        assert_eq!(
            extract_cached_english_markdown(&raw, &source, Some("fr")).unwrap().map(|cache| cache.markdown),
            None
        );
    }

    #[test]
    fn test_changed_summary_inputs_reject_cache() {
        let source = sample_cache_source();
        let template_fingerprint = source.template_fingerprint.clone();
        let raw = build_summary_result_json(
            "# Reunion\n## Points\nBonjour",
            "# Meeting\n## Points\nHello",
            source,
            Some("fr"),
            false,
            false,
            false,
        )
        .unwrap()
        .to_string();

        let changed_sources = [
            build_summary_cache_source(
                "changed transcript",
                "custom prompt",
                "standard_meeting",
                &template_fingerprint,
                Some(ContextBudget::for_ollama(8192)),
                "ollama",
                "gemma3:1b",
                Some("http://localhost:11434"),
                None,
                None,
                None,
                None,
            ),
            build_summary_cache_source(
                "transcript body",
                "changed prompt",
                "standard_meeting",
                &template_fingerprint,
                Some(ContextBudget::for_ollama(8192)),
                "ollama",
                "gemma3:1b",
                Some("http://localhost:11434"),
                None,
                None,
                None,
                None,
            ),
            build_summary_cache_source(
                "transcript body",
                "custom prompt",
                "daily_standup",
                &template_fingerprint,
                Some(ContextBudget::for_ollama(8192)),
                "ollama",
                "gemma3:1b",
                Some("http://localhost:11434"),
                None,
                None,
                None,
                None,
            ),
            build_summary_cache_source(
                "transcript body",
                "custom prompt",
                "standard_meeting",
                &template_fingerprint,
                Some(ContextBudget::for_ollama(8192)),
                "openai",
                "gemma3:1b",
                Some("http://localhost:11434"),
                None,
                None,
                None,
                None,
            ),
            build_summary_cache_source(
                "transcript body",
                "custom prompt",
                "standard_meeting",
                &template_fingerprint,
                Some(ContextBudget::for_ollama(8192)),
                "ollama",
                "qwen2.5:3b",
                Some("http://localhost:11434"),
                None,
                None,
                None,
                None,
            ),
            build_summary_cache_source(
                "transcript body",
                "custom prompt",
                "standard_meeting",
                &template_fingerprint,
                Some(ContextBudget::for_ollama(8192)),
                "ollama",
                "gemma3:1b",
                Some("http://localhost:11500"),
                None,
                None,
                None,
                None,
            ),
            build_summary_cache_source(
                "transcript body",
                "custom prompt",
                "standard_meeting",
                &template_fingerprint,
                Some(ContextBudget::for_ollama(8192)),
                "ollama",
                "gemma3:1b",
                Some("http://localhost:11434"),
                Some("https://custom.example/v1"),
                Some(2048),
                Some(0.2),
                Some(0.9),
            ),
        ];

        for changed_source in changed_sources {
            assert_eq!(
                extract_cached_english_markdown(&raw, &changed_source, Some("de")).unwrap().map(|cache| cache.markdown),
                None
            );
        }
    }

    #[test]
    fn test_changed_template_content_rejects_cache() {
        let source = sample_cache_source();
        let raw = build_summary_result_json(
            "# Reunion\n## Points\nBonjour",
            "# Meeting\n## Points\nHello",
            source.clone(),
            Some("fr"),
            false,
            false,
            false,
        )
        .unwrap()
        .to_string();

        let changed_template = SummaryCacheSource {
            template_fingerprint: stable_text_fingerprint("changed template prompt"),
            ..source
        };

        assert_eq!(
            extract_cached_english_markdown(&raw, &changed_template, Some("de")).unwrap().map(|cache| cache.markdown),
            None
        );
    }

    #[test]
    fn test_changed_context_budget_rejects_cache() {
        let source = sample_cache_source();
        let raw = build_summary_result_json(
            "# Reunion\n## Points\nBonjour",
            "# Meeting\n## Points\nHello",
            source.clone(),
            Some("fr"),
            false,
            false,
            false,
        )
        .unwrap()
        .to_string();

        let changed_budget = SummaryCacheSource {
            context_budget: Some(ContextBudget::for_ollama(16_384)),
            ..source
        };

        assert_eq!(
            extract_cached_english_markdown(&raw, &changed_budget, Some("de")).unwrap().map(|cache| cache.markdown),
            None
        );
    }

    #[test]
    fn cache_rows_from_before_context_budgets_miss_for_ollama() {
        // Rows written before budgets carried `token_threshold` (context minus 300) instead.
        let source = sample_cache_source();
        let mut raw = build_summary_result_json(
            "# Reunion\n## Points\nBonjour",
            "# Meeting\n## Points\nHello",
            source.clone(),
            Some("fr"),
            false,
            false,
            false,
        )
        .unwrap();
        let legacy_source = raw["english_cache"]["source"].as_object_mut().unwrap();
        legacy_source.remove("context_budget");
        legacy_source.insert("token_threshold".to_string(), serde_json::json!(7892));

        assert_eq!(
            extract_cached_english_markdown(&raw.to_string(), &source, Some("de")).unwrap().map(|cache| cache.markdown),
            None
        );
    }

    #[test]
    fn test_result_json_strips_display_markdown_but_keeps_cache_title() {
        let result = build_summary_result_json(
            "# Translated Title\n## Decisions\nDone",
            "# English Title\n## Decisions\nDone",
            sample_cache_source(),
            Some("fr"),
            false,
            false,
            false,
        )
        .unwrap();

        assert_eq!(result["markdown"], "## Decisions\nDone");
        assert_eq!(
            result["english_cache"]["markdown"],
            "# English Title\n## Decisions\nDone"
        );
    }

    #[test]
    fn result_json_rejects_title_only_display_markdown() {
        assert_eq!(
            build_summary_result_json(
                "# Title",
                "# Title",
                sample_cache_source(),
                None,
                false,
                false,
                false,
            ),
            Err("Final summary contains no visible content after title removal".to_string())
        );
    }

    #[test]
    fn result_json_never_persists_reasoning_text() {
        let result = build_summary_result_json(
            "# Title\nVisible",
            "# Title\nVisible",
            sample_cache_source(),
            None,
            true,
            false,
            false,
        )
        .unwrap();
        assert_eq!(result["reasoning_stripped"], true);
        assert!(result.get("reasoning").is_none());
    }

    #[test]
    fn result_json_records_a_cut_off_combine() {
        let result = build_summary_result_json(
            "# Title\nVisible",
            "# Title\nVisible",
            sample_cache_source(),
            None,
            false,
            false,
            true,
        )
        .unwrap();
        assert_eq!(result["combine_truncated"], true);
        assert_eq!(result["normalization_fallback"], false);
    }

    #[test]
    fn a_cut_off_combine_survives_a_cached_english_rerun() {
        let source = sample_cache_source();
        let mut raw = build_summary_result_json(
            "# Reunion\n## Points\nBonjour",
            "# Meeting\n## Points\nHello",
            source.clone(),
            Some("fr"),
            false,
            false,
            true,
        )
        .unwrap();

        let cached = extract_cached_english_markdown(&raw.to_string(), &source, Some("de"))
            .unwrap()
            .expect("cache hit");
        assert!(cached.combine_truncated);

        // Rows written before the flag existed read as not truncated.
        raw["english_cache"].as_object_mut().unwrap().remove("combine_truncated");
        let legacy = extract_cached_english_markdown(&raw.to_string(), &source, Some("de"))
            .unwrap()
            .expect("cache hit");
        assert!(!legacy.combine_truncated);
    }

    #[test]
    fn test_extract_cached_english_from_malformed_json_errors() {
        let raw = r#"{ not valid json"#;
        assert!(extract_cached_english_markdown(raw, &sample_cache_source(), Some("de")).is_err());
    }

    #[tokio::test]
    async fn openrouter_budget_comes_from_the_catalogue_context_length() {
        use crate::summary::llm_client::test_http::{read_http_request, write_json_response};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/api/v1/models", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            read_http_request(&mut stream).await;
            let body = br#"{"data":[{"id":"mistralai/mistral-7b-instruct","context_length":32768,"top_provider":{"context_length":8192}}]}"#;
            write_json_response(&mut stream, body).await;
        });

        let budget = SummaryService::select_context_budget(
            &LLMProvider::OpenRouter,
            "mistralai/mistral-7b-instruct",
            None,
            &url,
        )
        .await;
        server.await.unwrap();

        assert_eq!(budget, Ok(Some(ContextBudget::for_hosted_model(8192))));
    }

    #[tokio::test]
    async fn openrouter_without_a_catalogue_and_other_cloud_providers_stay_single_pass() {
        // Port 9 (discard) on localhost refuses the connection immediately.
        let unreachable = "http://127.0.0.1:9/api/v1/models";
        assert_eq!(
            SummaryService::select_context_budget(&LLMProvider::OpenRouter, "any/model", None, unreachable)
                .await,
            Ok(None)
        );
        for provider in [
            LLMProvider::OpenAI,
            LLMProvider::Claude,
            LLMProvider::Groq,
            LLMProvider::CustomOpenAI,
        ] {
            assert_eq!(
                SummaryService::select_context_budget(&provider, "model", None, unreachable).await,
                Ok(None)
            );
        }
    }

    #[tokio::test]
    async fn builtin_budget_comes_from_the_model_registry() {
        let unused = "http://127.0.0.1:9/";
        assert_eq!(
            SummaryService::select_context_budget(&LLMProvider::BuiltInAI, "qwen3.5:4b", None, unused)
                .await,
            Ok(Some(ContextBudget::for_builtin(32_768, models::DEFAULT_MAX_TOKENS)))
        );
        assert!(
            SummaryService::select_context_budget(&LLMProvider::BuiltInAI, "nope", None, unused)
                .await
                .is_err()
        );
    }
}
