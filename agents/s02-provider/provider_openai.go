package main

// OpenAI Provider: stdlib net/http only, with retry-with-backoff for the
// transient 429 / 5xx errors. Mirrors upstream `openai_complete_if_cache`
// in `lightrag/llm/openai.py`, minus the AsyncOpenAI SDK and minus the
// keyword-extraction structured output (we do plain chat completion).
//
// Key teaching points:
//   - retry uses the SAME stdlib pattern (sleep + jitter) as upstream's
//     tenacity decorator: stop_after_attempt(3) + exponential backoff.
//   - context.Context cancellation is honored on every wait (s01 never
//     waited; s02's retry loop is the first place where `ctx.Done()`
//     truly matters at request time).
//   - functional options pattern is used INSTEAD of a 10-field constructor,
//     because Phase G will keep adding fields (proxy, baseURL alts, …).

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"time"
)

// OpenAIProvider holds connection state. APIKey is read at construction
// (so a single process can swap keys with WithAPIKey on a fresh provider).
type OpenAIProvider struct {
	APIKey     string
	Model      string
	BaseURL    string
	HTTP       *http.Client
	MaxRetries int    // total attempts is MaxRetries + 1 ; default = 2 retries i.e. 3 total
	Logger     Logger // structured printf; noop by default
}

// NewOpenAIProvider applies functional options on top of sane defaults.
// The defaults match s01: gpt-4o-mini, 60s timeout, the public OpenAI URL,
// API key from env, no logger. With MaxRetries = 2 we get 3 total attempts.
func NewOpenAIProvider(opts ...ProviderOption) *OpenAIProvider {
	p := &OpenAIProvider{
		APIKey:     os.Getenv("OPENAI_API_KEY"),
		Model:      "gpt-4o-mini",
		BaseURL:    "https://api.openai.com/v1",
		HTTP:       &http.Client{Timeout: 60 * time.Second},
		MaxRetries: 2,
		Logger:     noopLogger,
	}
	for _, opt := range opts {
		opt(p)
	}
	return p
}

// openAIChatPayload mirrors the documented chat-completions schema. We
// intentionally do NOT pull in the official SDK — stdlib net/http only,
// for transparency and zero-dep CI.
type openAIChatPayload struct {
	Model       string              `json:"model"`
	Messages    []openAIChatMessage `json:"messages"`
	MaxTokens   int                 `json:"max_tokens,omitempty"`
	Temperature float64             `json:"temperature,omitempty"`
	Stream      bool                `json:"stream"`
}

type openAIChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type openAIChatResponse struct {
	Choices []struct {
		Message openAIChatMessage `json:"message"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error,omitempty"`
}

// transientStatusError represents a 429 or 5xx that the retry loop should
// recover from. Distinguishing transient vs. permanent is the SAME idea as
// upstream's tenacity `retry_if_exception_type(RateLimitError | …)`.
type transientStatusError struct {
	Status int
	Body   string
}

func (e *transientStatusError) Error() string {
	return fmt.Sprintf("openai: transient status %d: %s", e.Status, e.Body)
}

// Complete is the single-method Provider implementation. It builds the
// payload once and retries the HTTP round-trip up to MaxRetries times on
// transient failures. Token usage flows from the response `usage` field
// into CompleteResponse.
func (p *OpenAIProvider) Complete(ctx context.Context, req CompleteRequest) (CompleteResponse, error) {
	if p.APIKey == "" {
		return CompleteResponse{}, errors.New("openai: OPENAI_API_KEY is empty (set it or use MockProvider)")
	}
	model := req.Model
	if model == "" {
		model = p.Model
	}

	// Build the messages array: optional system, then the user/assistant
	// history. Empty system is omitted to match upstream behavior.
	msgs := make([]openAIChatMessage, 0, len(req.Messages)+1)
	if req.System != "" {
		msgs = append(msgs, openAIChatMessage{Role: "system", Content: req.System})
	}
	for _, m := range req.Messages {
		msgs = append(msgs, openAIChatMessage{Role: m.Role, Content: m.Content})
	}
	body, err := json.Marshal(openAIChatPayload{
		Model: model, Messages: msgs,
		MaxTokens: req.MaxTokens, Temperature: req.Temperature,
		Stream: false, // s02 forces non-stream; s11 / Phase G enables true.
	})
	if err != nil {
		return CompleteResponse{}, fmt.Errorf("openai: marshal: %w", err)
	}

	url := p.BaseURL + "/chat/completions"

	// Retry loop: the same body is replayed on each attempt. The first
	// attempt is "attempt 0", retries are 1..MaxRetries. We wait 200ms,
	// 600ms, 1.8s before retries 1, 2, 3… (3x exponential) plus a small
	// jitter so concurrent retries don't synchronize.
	var lastErr error
	for attempt := 0; attempt <= p.MaxRetries; attempt++ {
		if attempt > 0 {
			wait := backoffDelay(attempt)
			p.Logger("openai: retry attempt=%d wait=%s lastErr=%v", attempt, wait, lastErr)
			select {
			case <-ctx.Done():
				return CompleteResponse{}, fmt.Errorf("openai: cancelled while waiting to retry: %w", ctx.Err())
			case <-time.After(wait):
			}
		}
		resp, err := p.doOnce(ctx, url, body)
		if err == nil {
			p.Logger("openai: success attempt=%d input_tokens=%d output_tokens=%d",
				attempt, resp.InputTokens, resp.OutputTokens)
			return resp, nil
		}
		lastErr = err
		// Only loop on transient status errors. Network errors and 4xx
		// (other than 429) are surfaced immediately.
		var tse *transientStatusError
		if !errors.As(err, &tse) {
			return CompleteResponse{}, err
		}
	}
	return CompleteResponse{}, fmt.Errorf("openai: exhausted %d retries: %w", p.MaxRetries, lastErr)
}

// doOnce performs a single HTTP round-trip and decodes the response. It
// returns *transientStatusError for retry-eligible failures and a regular
// error for permanent ones (auth, malformed JSON, …).
func (p *OpenAIProvider) doOnce(ctx context.Context, url string, body []byte) (CompleteResponse, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return CompleteResponse{}, fmt.Errorf("openai: build req: %w", err)
	}
	httpReq.Header.Set("Authorization", "Bearer "+p.APIKey)
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := p.HTTP.Do(httpReq)
	if err != nil {
		// network-level error; ctx cancellation surfaces here as wrapped error
		return CompleteResponse{}, fmt.Errorf("openai: do: %w", err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)

	// Permanent failures: bail out with a useful, non-retried error.
	if resp.StatusCode == http.StatusUnauthorized {
		return CompleteResponse{}, fmt.Errorf("openai: 401 unauthorized — check OPENAI_API_KEY")
	}
	// Transient: 429 or 5xx. Returns a typed error the retry loop matches on.
	if resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500 {
		return CompleteResponse{}, &transientStatusError{Status: resp.StatusCode, Body: string(raw)}
	}
	if resp.StatusCode != http.StatusOK {
		return CompleteResponse{}, fmt.Errorf("openai: status=%d body=%s", resp.StatusCode, string(raw))
	}

	var out openAIChatResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return CompleteResponse{}, fmt.Errorf("openai: decode: %w (body=%s)", err, string(raw))
	}
	if out.Error != nil {
		return CompleteResponse{}, fmt.Errorf("openai: %s: %s", out.Error.Type, out.Error.Message)
	}
	if len(out.Choices) == 0 {
		return CompleteResponse{}, errors.New("openai: no choices in response")
	}
	return CompleteResponse{
		Text:         out.Choices[0].Message.Content,
		InputTokens:  out.Usage.PromptTokens,
		OutputTokens: out.Usage.CompletionTokens,
	}, nil
}

// backoffDelay returns the wait duration before the Nth retry (1-based).
// Pattern: 200ms, 600ms, 1.8s, … (each step is 3x). A small uniform jitter
// of ±20% prevents thundering-herd retries when many goroutines hit a 429
// at once.
func backoffDelay(attempt int) time.Duration {
	base := 200 * time.Millisecond
	mult := 1
	for i := 1; i < attempt; i++ {
		mult *= 3
	}
	d := time.Duration(mult) * base
	jitter := time.Duration(rand.Int63n(int64(d) / 5)) // up to ±20%
	if rand.Intn(2) == 0 {
		return d - jitter
	}
	return d + jitter
}
