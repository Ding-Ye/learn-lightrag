package main

// OpenAIEmbedder — stdlib net/http only, no SDK. Mirrors upstream
// `lightrag/llm/openai.py:openai_embed` (lines 733-895) but trims the
// Azure / token-tracker / max_token_size truncation knobs so s06 stays
// teachable in ~200 LOC. Keeps the load-bearing primitives:
//
//   - Batched POST: one HTTP call carries up to BatchSize texts; the
//     response.data[i] aligns with input[i].
//   - 429/5xx retry with exponential backoff (matches upstream's
//     `@retry(stop_after_attempt(3), wait_exponential(...))`).
//   - Functional-options ctor (matches s02's pattern; re-implemented here
//     because sessions can't cross-import).
//   - Dim() introspection (matches upstream's @wrap_embedding_func_with_attrs
//     `embedding_dim=1536` decoration on the function object).

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

// OpenAIEmbedder is the production embedder. Default values mirror the
// upstream defaults so a learner reading both side-by-side recognises
// every constant.
type OpenAIEmbedder struct {
	APIKey     string
	Model      string
	BaseURL    string
	BatchSize  int
	MaxRetries int
	HTTP       *http.Client
	dim        int // Dim() returns this; tied to Model. Defaults to 1536.
}

// NewOpenAIEmbedder applies functional options on top of upstream-aligned
// defaults: gpt-/text-embedding-3-small, BaseURL=public OpenAI, BatchSize=128
// (~conservative vs the documented ~2048 cap), MaxRetries=3 (matches
// stop_after_attempt(3)), 60s HTTP timeout, dim=1536.
func NewOpenAIEmbedder(opts ...EmbedderOption) *OpenAIEmbedder {
	e := &OpenAIEmbedder{
		APIKey:     os.Getenv("OPENAI_API_KEY"),
		Model:      "text-embedding-3-small",
		BaseURL:    "https://api.openai.com/v1",
		BatchSize:  128,
		MaxRetries: 3,
		HTTP:       &http.Client{Timeout: 60 * time.Second},
		dim:        1536,
	}
	for _, opt := range opts {
		opt(e)
	}
	return e
}

// Dim returns the per-vector dimension. s07's VectorStore reads this at
// construction time to size its index. Hard-coded to 1536 for s06; Phase G
// will look this up by model name.
func (e *OpenAIEmbedder) Dim() int { return e.dim }

// embedRequest is the documented `/v1/embeddings` request schema. We send
// `encoding_format: float` for transparency (vs upstream's default base64
// which saves wire size at the cost of two extra base64 decode steps).
type embedRequest struct {
	Model          string   `json:"model"`
	Input          []string `json:"input"`
	EncodingFormat string   `json:"encoding_format,omitempty"`
}

// embedResponse mirrors what `embeddings.create` returns. We only consume
// data[].embedding; the upstream code also reads usage.{prompt,total}_tokens
// when token_tracker is set — s06 omits that wiring.
type embedResponse struct {
	Data []struct {
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error,omitempty"`
}

// Embed slices texts into BatchSize-sized batches, posts each to the
// embeddings endpoint with retry, and reassembles results in input order.
//
// Empty input is the load-bearing edge case: it must return a non-nil
// empty slice with no HTTP call. Doc-status pipelines call Embed even for
// docs that yield zero chunks (titles only, etc.) and we should not burn
// quota on an empty list.
func (e *OpenAIEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return [][]float32{}, nil
	}
	if e.APIKey == "" {
		return nil, errors.New("openai-embed: OPENAI_API_KEY is empty (set it or use MockEmbedder)")
	}
	if e.BatchSize <= 0 {
		return nil, fmt.Errorf("openai-embed: invalid BatchSize=%d", e.BatchSize)
	}

	out := make([][]float32, 0, len(texts))
	for start := 0; start < len(texts); start += e.BatchSize {
		end := start + e.BatchSize
		if end > len(texts) {
			end = len(texts)
		}
		var batchVecs [][]float32
		err := withRetry(e.MaxRetries, func() error {
			v, err := e.embedOnce(ctx, texts[start:end])
			if err != nil {
				return err
			}
			batchVecs = v
			return nil
		})
		if err != nil {
			return nil, err
		}
		out = append(out, batchVecs...)
	}
	return out, nil
}

// embedOnce performs ONE HTTP round-trip for ONE batch. Returns
// *transientStatusError on 429/5xx so withRetry can sleep and replay; returns
// regular errors on permanent failures (auth, JSON decode, …).
func (e *OpenAIEmbedder) embedOnce(ctx context.Context, batch []string) ([][]float32, error) {
	body, err := json.Marshal(embedRequest{
		Model:          e.Model,
		Input:          batch,
		EncodingFormat: "float",
	})
	if err != nil {
		return nil, fmt.Errorf("openai-embed: marshal: %w", err)
	}

	url := e.BaseURL + "/embeddings"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai-embed: build req: %w", err)
	}
	httpReq.Header.Set("Authorization", "Bearer "+e.APIKey)
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := e.HTTP.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("openai-embed: do: %w", err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)

	// Permanent failures: surface immediately, no retry.
	if resp.StatusCode == http.StatusUnauthorized {
		return nil, fmt.Errorf("openai-embed: 401 unauthorized — check OPENAI_API_KEY")
	}
	// Transient: 429 or 5xx → typed error withRetry recognises.
	if resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500 {
		return nil, &transientStatusError{Status: resp.StatusCode, Body: string(raw)}
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("openai-embed: status=%d body=%s", resp.StatusCode, string(raw))
	}

	var out embedResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("openai-embed: decode: %w (body=%s)", err, string(raw))
	}
	if out.Error != nil {
		return nil, fmt.Errorf("openai-embed: %s: %s", out.Error.Type, out.Error.Message)
	}
	if len(out.Data) != len(batch) {
		return nil, fmt.Errorf("openai-embed: returned %d vectors for %d inputs", len(out.Data), len(batch))
	}

	// OpenAI documents that response.data[i].index == request.input[i],
	// but we sort by Index defensively — if any provider reorders, we
	// still produce input-aligned output. (Mock httptest servers in s06
	// preserve order anyway; this guards future-Phase-G alternative
	// providers reusing the same endpoint shape.)
	vecs := make([][]float32, len(batch))
	for _, dp := range out.Data {
		if dp.Index < 0 || dp.Index >= len(batch) {
			return nil, fmt.Errorf("openai-embed: out-of-range index %d for batch len %d", dp.Index, len(batch))
		}
		vecs[dp.Index] = dp.Embedding
	}
	for i, v := range vecs {
		if v == nil {
			return nil, fmt.Errorf("openai-embed: missing vector at index %d", i)
		}
	}
	return vecs, nil
}
