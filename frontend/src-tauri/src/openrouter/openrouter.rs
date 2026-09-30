use serde::{Deserialize, Serialize};
use std::time::Duration;
use tauri::command;

/// OpenRouter's public model catalogue.
pub const MODELS_URL: &str = "https://openrouter.ai/api/v1/models";

/// How long a summary waits for the catalogue before running without a context budget.
const CATALOGUE_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenRouterModel {
    pub id: String,
    pub name: String,
    pub context_length: Option<u32>,
    pub prompt_price: Option<String>,
    pub completion_price: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterApiModel {
    id: String,
    name: Option<String>,
    context_length: Option<u32>,
    #[serde(default)]
    top_provider: Option<TopProvider>,
    #[serde(default)]
    pricing: Option<Pricing>,
}

impl OpenRouterApiModel {
    /// The window of the provider OpenRouter routes to first, else the model's own.
    fn effective_context_length(&self) -> Option<u32> {
        self.top_provider
            .as_ref()
            .and_then(|tp| tp.context_length)
            .or(self.context_length)
    }
}

#[derive(Debug, Deserialize, Default)]
struct TopProvider {
    context_length: Option<u32>,
}

#[derive(Debug, Deserialize, Default)]
struct Pricing {
    prompt: Option<String>,
    completion: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterResponse {
    data: Vec<OpenRouterApiModel>,
}

#[command]
pub fn get_openrouter_models() -> Result<Vec<OpenRouterModel>, String> {
    let client = reqwest::blocking::Client::new();
    let response = client
        .get(MODELS_URL)
        .send()
        .map_err(|e| format!("Failed to make HTTP request: {}", e))?;

    if !response.status().is_success() {
        return Err(format!("HTTP request failed with status: {}", response.status()));
    }

    let api_response: OpenRouterResponse = response
        .json()
        .map_err(|e| format!("Failed to parse JSON response: {}", e))?;

    let models = api_response
        .data
        .into_iter()
        .map(|m| OpenRouterModel {
            context_length: m.effective_context_length(),
            id: m.id,
            name: m.name.unwrap_or_else(|| "Unknown".to_string()),
            prompt_price: m.pricing.as_ref().and_then(|p| p.prompt.clone()),
            completion_price: m.pricing.as_ref().and_then(|p| p.completion.clone()),
        })
        .collect();

    Ok(models)
}

/// Context window of `model_id` in the catalogue at `models_url` (normally [`MODELS_URL`]).
/// `Ok(None)` when the model is not listed or reports no context length.
pub async fn fetch_context_length(
    client: &reqwest::Client,
    models_url: &str,
    model_id: &str,
) -> Result<Option<u32>, String> {
    let response = client
        .get(models_url)
        .timeout(CATALOGUE_TIMEOUT)
        .send()
        .await
        .map_err(|e| format!("Failed to fetch the OpenRouter model list: {}", e))?;
    if !response.status().is_success() {
        return Err(format!(
            "OpenRouter model list request failed with status: {}",
            response.status()
        ));
    }
    let catalogue: OpenRouterResponse = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse the OpenRouter model list: {}", e))?;
    Ok(catalogue
        .data
        .iter()
        .find(|model| model.id == model_id)
        .and_then(OpenRouterApiModel::effective_context_length))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::summary::llm_client::test_http::{read_http_request, write_json_response};

    async fn serve_catalogue(body: &'static str) -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/api/v1/models", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            read_http_request(&mut stream).await;
            write_json_response(&mut stream, body.as_bytes()).await;
        });
        (url, server)
    }

    const CATALOGUE: &str = r#"{"data":[
        {"id":"meta-llama/llama-3-8b-instruct","name":"Llama 3 8B","context_length":8192,
         "top_provider":{"context_length":8192}},
        {"id":"routed/model","context_length":131072,"top_provider":{"context_length":32768}},
        {"id":"own/model","context_length":65536,"top_provider":{"context_length":null}},
        {"id":"unknown/window"}
    ]}"#;

    #[tokio::test]
    async fn context_length_prefers_the_top_provider_window() {
        let client = reqwest::Client::new();
        for (model, expected) in [
            ("meta-llama/llama-3-8b-instruct", Some(8192)),
            ("routed/model", Some(32768)),
            ("own/model", Some(65536)),
            ("unknown/window", None),
            ("missing/model", None),
        ] {
            let (url, server) = serve_catalogue(CATALOGUE).await;
            assert_eq!(fetch_context_length(&client, &url, model).await, Ok(expected), "{model}");
            server.await.unwrap();
        }
    }
}
