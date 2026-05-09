package main

// Provider abstraction for the LLM (chat completion). Mirrors upstream
// `llm_model_func: Callable` injection at lightrag/lightrag.py around the
// dataclass field declarations (see upstream-readings/s01-lightrag.py).
//
// In s01 we ship one real implementation (OpenAIProvider) plus a deterministic
// MockProvider for tests / offline CI. Phase G later adds Anthropic, Bedrock,
// Ollama profiles — none of those touch the interface shape below.

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

// Provider is the single-method LLM contract every chapter inherits.
// Streaming is reserved for s11 + Phase G; s01..s10 always pass Stream=false.
type Provider interface {
	Complete(ctx context.Context, req CompleteRequest) (CompleteResponse, error)
}

type CompleteRequest struct {
	Model       string
	System      string
	Messages    []Message // user/assistant turns, oldest first
	MaxTokens   int
	Temperature float64
	Stream      bool
}

type Message struct {
	Role    string // "user" | "assistant" | "system"
	Content string
}

type CompleteResponse struct {
	Text         string
	InputTokens  int
	OutputTokens int
}

// OpenAIProvider hits https://api.openai.com/v1/chat/completions with stdlib
// net/http only — no SDK. Reads OPENAI_API_KEY at call time so the same
// process can swap keys.
type OpenAIProvider struct {
	APIKey string
	Model  string // default "gpt-4o-mini"
	HTTP   *http.Client
}

// NewOpenAIProvider returns a provider that reads OPENAI_API_KEY from env.
func NewOpenAIProvider() *OpenAIProvider {
	return &OpenAIProvider{
		APIKey: os.Getenv("OPENAI_API_KEY"),
		Model:  "gpt-4o-mini",
		HTTP:   &http.Client{Timeout: 60 * time.Second},
	}
}

// openAIChatPayload mirrors the documented chat-completions schema.
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

func (p *OpenAIProvider) Complete(ctx context.Context, req CompleteRequest) (CompleteResponse, error) {
	if p.APIKey == "" {
		return CompleteResponse{}, errors.New("openai: OPENAI_API_KEY is empty; export it or use -provider mock")
	}
	model := req.Model
	if model == "" {
		model = p.Model
	}
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
		Stream: false,
	})
	if err != nil {
		return CompleteResponse{}, fmt.Errorf("openai: marshal: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		"https://api.openai.com/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return CompleteResponse{}, fmt.Errorf("openai: build req: %w", err)
	}
	httpReq.Header.Set("Authorization", "Bearer "+p.APIKey)
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := p.HTTP.Do(httpReq)
	if err != nil {
		return CompleteResponse{}, fmt.Errorf("openai: do: %w", err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode == 401 {
		return CompleteResponse{}, fmt.Errorf("openai: 401 unauthorized — check OPENAI_API_KEY")
	}
	if resp.StatusCode == 429 {
		return CompleteResponse{}, fmt.Errorf("openai: 429 rate-limited — retry later or use -provider mock")
	}
	if resp.StatusCode >= 500 {
		return CompleteResponse{}, fmt.Errorf("openai: 5xx upstream error: %s", string(raw))
	}
	if resp.StatusCode != 200 {
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
