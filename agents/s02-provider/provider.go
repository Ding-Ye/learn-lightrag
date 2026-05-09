package main

// Provider abstraction for the LLM (chat completion). Mirrors upstream
// `llm_model_func: Callable` in `lightrag/lightrag.py` (see
// upstream-readings/s02-openai.py for the verbatim slice).
//
// In s01 the OpenAI HTTP call lived inline next to the Provider interface.
// s02's job is to PROMOTE Provider to a real chapter: same one-method
// contract, three concrete impls (real OpenAI, deterministic Mock, Echo
// for trivial smoke), functional-option construction, and retry-with-
// backoff in the OpenAI impl. The interface shape itself stays frozen
// from s01 onward — Phase G adds Anthropic / Bedrock / Ollama next door
// without changing CompleteRequest / CompleteResponse / Provider.

import (
	"context"
	"fmt"
	"time"
)

// Provider is the single-method LLM contract every chapter inherits.
// Streaming is reserved for s11 + Phase G; s01..s10 always pass Stream=false.
type Provider interface {
	Complete(ctx context.Context, req CompleteRequest) (CompleteResponse, error)
}

// CompleteRequest carries one chat-completion turn-set. Same shape as s01.
// The fields stay stable so future provider implementations can be added in
// Phase G without breaking call sites.
type CompleteRequest struct {
	Model       string
	System      string
	Messages    []Message // user/assistant turns, oldest first
	MaxTokens   int
	Temperature float64
	Stream      bool
}

// Message is one turn — Role is "user" | "assistant" | "system".
type Message struct {
	Role    string
	Content string
}

// CompleteResponse mirrors what we get back from the LLM. s02 actually
// populates InputTokens / OutputTokens (s01 left them as best-effort).
type CompleteResponse struct {
	Text         string
	InputTokens  int
	OutputTokens int
}

// Logger is a tiny printf-style sink so Provider impls can record retry
// attempts and token counts WITHOUT pulling in log/slog as a hard dep.
// Any function with the signature works (including the stdlib `log.Printf`
// when wrapped). Pass a no-op (`func(string, ...any) {}`) to silence.
type Logger func(format string, args ...any)

// noopLogger discards everything. Used when the caller doesn't supply one.
func noopLogger(format string, args ...any) {}

// ProviderOption mutates an OpenAIProvider during construction. We use the
// functional-option pattern (vs. a giant config struct or many ctor args)
// because Phase G keeps adding optional knobs and we want each addition to
// be one new `WithFoo()` exported function — no breaking change at the
// call sites.
type ProviderOption func(*OpenAIProvider)

// WithModel pins the OpenAI model name (default "gpt-4o-mini").
func WithModel(model string) ProviderOption {
	return func(p *OpenAIProvider) { p.Model = model }
}

// WithAPIKey sets the API key. If unset, the provider falls back to
// reading OPENAI_API_KEY at construction time (NewOpenAIProvider does
// the env lookup).
func WithAPIKey(key string) ProviderOption {
	return func(p *OpenAIProvider) { p.APIKey = key }
}

// WithBaseURL overrides the OpenAI endpoint. The default is
// "https://api.openai.com/v1"; tests use httptest.NewServer's URL, and
// users behind a corp proxy / Azure deployment / vLLM-compatible server
// also override here.
func WithBaseURL(baseURL string) ProviderOption {
	return func(p *OpenAIProvider) { p.BaseURL = baseURL }
}

// WithTimeout sets the per-request HTTP timeout (default 60s).
func WithTimeout(d time.Duration) ProviderOption {
	return func(p *OpenAIProvider) {
		if p.HTTP == nil {
			return
		}
		p.HTTP.Timeout = d
	}
}

// WithMaxRetries sets the retry attempt count for transient failures
// (429 / 5xx). Default is 3 (i.e. 1 initial try + up to 2 retries =
// 3 total attempts). Set to 0 for no retry — useful in tests that want
// to assert "first error reaches caller".
func WithMaxRetries(n int) ProviderOption {
	return func(p *OpenAIProvider) { p.MaxRetries = n }
}

// WithLogger swaps the structured logger. Default is a noop. Tests use a
// closure that captures lines into a slice for assertion.
func WithLogger(logger Logger) ProviderOption {
	return func(p *OpenAIProvider) { p.Logger = logger }
}

// errString is a sentinel error helper that pre-formats once. Used for the
// retry path so we don't allocate on every attempt.
func errString(format string, a ...any) error { return fmt.Errorf(format, a...) }
