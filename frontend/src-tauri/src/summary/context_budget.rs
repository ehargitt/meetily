//! Token estimates and context-window budgets for chunked summary generation.
//!
//! A request must fit the model's context window with room left for the completion. The same
//! [`ContextBudget`] decides how large a transcript chunk may be and, for Ollama, which `num_ctx`
//! the request asks for, so the two can never disagree.

use serde::{Deserialize, Serialize};

/// Estimates are counted in 1/20ths of a token so ASCII text keeps the historical rate of 0.35
/// tokens per character (7 units) exactly.
pub(crate) const UNITS_PER_TOKEN: usize = 20;
const ASCII_CHAR_UNITS: usize = 7;
/// Byte-level tokenizers never emit more than one token per UTF-8 byte, and modern multilingual
/// ones (Llama 3, Qwen, Gemma) emit well under half that for Cyrillic, CJK or Indic text. 0.4
/// tokens per byte (0.8 for a 2-byte character, 1.2 for a 3-byte one) keeps estimates for
/// non-Latin transcripts above their real size, where 0.35 per character undercounted CJK
/// roughly threefold.
const NON_ASCII_UNITS_PER_BYTE: usize = 8;

/// Largest context window requested from Ollama. Models report their trained maximum (often
/// 128k tokens), and asking for that makes Ollama allocate a KV cache far beyond typical VRAM;
/// 16k tokens covers a one to two hour meeting in a handful of chunks.
pub const OLLAMA_MAX_NUM_CTX: usize = 16_384;
/// Context assumed when Ollama's model metadata cannot be read.
pub const OLLAMA_FALLBACK_CONTEXT: usize = 4096;
/// Completion room kept free in Ollama and hosted-model windows (a quarter of small windows).
const OUTPUT_RESERVE: usize = 4096;
/// Chat-template role markers that the prompt estimate does not see.
const CHAT_TEMPLATE_SLACK: usize = 64;
/// Smallest transcript slice worth a request, even when the instructions crowd a tiny window.
const MIN_CONTENT_TOKENS: usize = 256;

/// Estimated token units of one character (see [`UNITS_PER_TOKEN`]).
pub(crate) fn char_token_units(c: char) -> usize {
    if c.is_ascii() {
        ASCII_CHAR_UNITS
    } else {
        NON_ASCII_UNITS_PER_BYTE * c.len_utf8()
    }
}

/// Conservative token estimate for any script; see [`NON_ASCII_UNITS_PER_BYTE`].
pub fn rough_token_count(s: &str) -> usize {
    s.chars()
        .map(char_token_units)
        .sum::<usize>()
        .div_ceil(UNITS_PER_TOKEN)
}

/// The context window a model runs with and the part of it kept free for the completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextBudget {
    /// Context window in tokens; sent to Ollama as `options.num_ctx`.
    pub context_tokens: usize,
    /// Tokens left free for the completion.
    pub output_reserve_tokens: usize,
}

impl ContextBudget {
    /// Budget for an Ollama model whose trained maximum is `model_max_context`, capped at
    /// [`OLLAMA_MAX_NUM_CTX`]. Small windows reserve a quarter for output instead of 4096.
    pub fn for_ollama(model_max_context: usize) -> Self {
        Self::for_hosted_model(model_max_context.min(OLLAMA_MAX_NUM_CTX))
    }

    /// Budget for a hosted model (e.g. on OpenRouter) whose window is `context_tokens`. No cap:
    /// the provider, not this machine, holds the KV cache.
    pub fn for_hosted_model(context_tokens: usize) -> Self {
        Self {
            context_tokens,
            output_reserve_tokens: OUTPUT_RESERVE.min(context_tokens / 4),
        }
    }

    /// Budget for a built-in model: its whole window, minus the `max_tokens` every request asks
    /// llama-helper to be able to generate.
    pub fn for_builtin(context_size: u32, max_tokens: i32) -> Self {
        Self {
            context_tokens: context_size as usize,
            output_reserve_tokens: max_tokens.max(0) as usize,
        }
    }

    /// Transcript tokens one request can carry next to `prompt_overhead_tokens` of instructions
    /// while leaving the output reserve free. Never below [`MIN_CONTENT_TOKENS`].
    pub fn content_tokens(&self, prompt_overhead_tokens: usize) -> usize {
        self.context_tokens
            .saturating_sub(self.output_reserve_tokens)
            .saturating_sub(prompt_overhead_tokens + CHAT_TEMPLATE_SLACK)
            .max(MIN_CONTENT_TOKENS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_estimate_keeps_the_historical_rate() {
        assert_eq!(rough_token_count(""), 0);
        assert_eq!(rough_token_count("a"), 1);
        assert_eq!(rough_token_count(&"a".repeat(100)), 35);
        assert_eq!(rough_token_count(&"a".repeat(1000)), 350);
    }

    #[test]
    fn non_latin_scripts_are_estimated_above_their_real_token_count() {
        // Real tokenizers produce roughly 0.7-1.0 tokens per CJK character and 0.4-0.5 per
        // Cyrillic one; the estimate must not fall below those.
        let japanese = "会議の内容を要約してください".repeat(10);
        let chars = japanese.chars().count();
        assert!(rough_token_count(&japanese) >= chars, "{} < {chars}", rough_token_count(&japanese));

        let russian = "Обсудили план выпуска".repeat(10);
        let letters = russian.chars().filter(|c| !c.is_ascii()).count();
        assert!(rough_token_count(&russian) * 2 >= letters);

        let hindi = "बैठक का सारांश".repeat(10);
        assert!(rough_token_count(&hindi) >= hindi.chars().count());
    }

    #[test]
    fn ollama_context_is_capped_and_reserves_output() {
        let budget = ContextBudget::for_ollama(131_072);
        assert_eq!(budget.context_tokens, OLLAMA_MAX_NUM_CTX);
        assert_eq!(budget.output_reserve_tokens, 4096);

        let small = ContextBudget::for_ollama(4096);
        assert_eq!(small.context_tokens, 4096);
        assert_eq!(small.output_reserve_tokens, 1024);
    }

    #[test]
    fn hosted_model_context_is_not_capped() {
        let budget = ContextBudget::for_hosted_model(200_000);
        assert_eq!(budget.context_tokens, 200_000);
        assert_eq!(budget.output_reserve_tokens, 4096);
        assert_eq!(ContextBudget::for_hosted_model(8192).output_reserve_tokens, 2048);
    }

    #[test]
    fn builtin_budget_reserves_max_tokens_and_prompt_overhead() {
        let budget = ContextBudget::for_builtin(32_768, 4096);
        assert_eq!(budget.content_tokens(1000), 32_768 - 4096 - 1000 - CHAT_TEMPLATE_SLACK);
        // Transcript, instructions and the full completion fit the window together.
        assert!(budget.content_tokens(1000) + 1000 + 4096 <= 32_768);
    }

    #[test]
    fn content_tokens_never_drops_below_the_floor() {
        let budget = ContextBudget::for_builtin(2048, 4096);
        assert_eq!(budget.content_tokens(500), MIN_CONTENT_TOKENS);
    }
}
