package main

// EmbeddingProvider — promoted in s06 from "inline call inside pipeline.go"
// to its own first-class chapter. Two implementations land in this session:
//   - OpenAIEmbedder: stdlib net/http POST /embeddings with batching + retry.
//   - MockEmbedder:   deterministic sha256 → 1536-d unit vector for CI.
//
// The interface itself was frozen in plan.md's shared types catalog. Phase G
// adds Anthropic / Bedrock / Cohere embedders by dropping more files next to
// these — no signature changes, no caller breakage.
//
// Mirrors upstream's `embedding_func: Callable[[list[str]], list[list[float]]]`
// shape, with `Dim()` corresponding to the `embedding_dim` attribute that
// `@wrap_embedding_func_with_attrs(embedding_dim=1536, ...)` attaches to the
// upstream function object in `lightrag/llm/openai.py:733-738`.

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"time"
)

// EmbeddingProvider is the single contract every embedder satisfies. Embed
// returns one [N]float32 vector per input text in INPUT ORDER (caller can
// zip texts[i] with result[i]). Dim returns the per-vector dimension; s07's
// VectorStore reads this at construction time to size its index.
type EmbeddingProvider interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
	Dim() int
}

// EmbedderOption mutates an OpenAIEmbedder during construction. We use
// functional options instead of a fat ctor so Phase G can keep adding knobs
// (proxy, encoding_format=base64, document/query prefix) as one new
// `WithFoo()` per addition without changing call sites.
type EmbedderOption func(*OpenAIEmbedder)

// WithModel pins the embedding model name. Default "text-embedding-3-small".
// Note: changing this should also change Dim() — but s06 keeps Dim hard-coded
// to 1536 for teaching simplicity. Phase G will tie Dim to the model lookup.
func WithModel(model string) EmbedderOption {
	return func(e *OpenAIEmbedder) { e.Model = model }
}

// WithAPIKey overrides the API key. Default reads OPENAI_API_KEY at
// construction time. Tests pass a literal key via httptest.
func WithAPIKey(key string) EmbedderOption {
	return func(e *OpenAIEmbedder) { e.APIKey = key }
}

// WithBaseURL overrides the OpenAI endpoint. Default "https://api.openai.com/v1".
// Tests use httptest.NewServer's URL; users behind Azure / vLLM / corp proxy
// also override here.
func WithBaseURL(baseURL string) EmbedderOption {
	return func(e *OpenAIEmbedder) { e.BaseURL = baseURL }
}

// WithBatchSize sets how many texts go into one HTTP call. Default 128 (a
// good middle ground: small enough to bound payload size and retry cost,
// big enough to amortize RTT). OpenAI's hard cap is ~2048; we stay
// conservative for teaching.
func WithBatchSize(n int) EmbedderOption {
	return func(e *OpenAIEmbedder) {
		if n > 0 {
			e.BatchSize = n
		}
	}
}

// WithMaxRetries sets transient-error retry budget. Default 3 (≈ 1 initial +
// 2 retries = 3 total attempts, matching upstream's `stop_after_attempt(3)`).
// Set 0 to disable retries — useful in tests asserting "first error reaches
// caller".
func WithMaxRetries(n int) EmbedderOption {
	return func(e *OpenAIEmbedder) {
		if n >= 0 {
			e.MaxRetries = n
		}
	}
}

// transientStatusError is the typed error withRetry detects to decide
// "should I sleep and try again?" 429 / 5xx are transient; 401 / 400 are
// permanent and surface immediately.
type transientStatusError struct {
	Status int
	Body   string
}

func (e *transientStatusError) Error() string {
	return fmt.Sprintf("openai-embed: transient status %d: %s", e.Status, e.Body)
}

// withRetry runs fn up to maxRetries+1 times, retrying ONLY when fn returns
// a *transientStatusError. Backoff is exponential 200ms → 600ms → 1.8s with
// ±20% jitter. Mirrors s02's pattern; re-implemented here in ~30 LOC because
// sessions cannot cross-import.
func withRetry(maxRetries int, fn func() error) error {
	var lastErr error
	for attempt := 0; attempt <= maxRetries; attempt++ {
		if attempt > 0 {
			time.Sleep(retryDelay(attempt))
		}
		err := fn()
		if err == nil {
			return nil
		}
		lastErr = err
		var tse *transientStatusError
		if !errors.As(err, &tse) {
			return err // permanent — bail out, no retry.
		}
	}
	return fmt.Errorf("openai-embed: exhausted %d retries: %w", maxRetries, lastErr)
}

// retryDelay returns the wait duration before the Nth retry (1-based).
// Pattern: 200ms, 600ms, 1.8s, … (each step is 3×). Adds ±20% jitter so
// concurrent retries do not synchronize on the same wakeup tick. We keep
// the base small (200ms) so test runtime stays bounded.
func retryDelay(attempt int) time.Duration {
	base := 200 * time.Millisecond
	mult := 1
	for i := 1; i < attempt; i++ {
		mult *= 3
	}
	d := time.Duration(mult) * base
	jitter := time.Duration(rand.Int63n(int64(d)/5 + 1))
	if rand.Intn(2) == 0 {
		return d - jitter
	}
	return d + jitter
}
