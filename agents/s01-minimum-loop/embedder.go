package main

// EmbeddingProvider mirrors upstream `embedding_func` (lightrag/lightrag.py
// dataclass field) — a function-shaped DI seam that takes a list of texts and
// returns a vector per text, plus a dimension hook.

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

// EmbeddingProvider is the minimal contract we need in s01.
type EmbeddingProvider interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
	Dim() int
}

// OpenAIEmbedder hits the embeddings endpoint with text-embedding-3-small.
// 1536-dim output. Reads OPENAI_API_KEY from env on construction.
type OpenAIEmbedder struct {
	APIKey string
	Model  string // default "text-embedding-3-small"
	HTTP   *http.Client
}

// NewOpenAIEmbedder builds an embedder using the env-default API key.
func NewOpenAIEmbedder() *OpenAIEmbedder {
	return &OpenAIEmbedder{
		APIKey: os.Getenv("OPENAI_API_KEY"),
		Model:  "text-embedding-3-small",
		HTTP:   &http.Client{Timeout: 60 * time.Second},
	}
}

// Dim is hard-coded to 1536, matching text-embedding-3-small.
// s06 makes this dynamic by introspecting one /v1/embeddings response.
func (o *OpenAIEmbedder) Dim() int { return 1536 }

type openAIEmbedPayload struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type openAIEmbedResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error,omitempty"`
}

func (o *OpenAIEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if o.APIKey == "" {
		return nil, errors.New("openai-embed: OPENAI_API_KEY is empty; export it or use -provider mock")
	}
	if len(texts) == 0 {
		return nil, nil
	}
	body, err := json.Marshal(openAIEmbedPayload{Model: o.Model, Input: texts})
	if err != nil {
		return nil, fmt.Errorf("openai-embed: marshal: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		"https://api.openai.com/v1/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai-embed: build req: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+o.APIKey)
	req.Header.Set("Content-Type", "application/json")
	resp, err := o.HTTP.Do(req)
	if err != nil {
		return nil, fmt.Errorf("openai-embed: do: %w", err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode == 401 {
		return nil, fmt.Errorf("openai-embed: 401 unauthorized — check OPENAI_API_KEY")
	}
	if resp.StatusCode == 429 {
		return nil, fmt.Errorf("openai-embed: 429 rate-limited — back off or use -provider mock")
	}
	if resp.StatusCode >= 500 {
		return nil, fmt.Errorf("openai-embed: 5xx upstream: %s", string(raw))
	}
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("openai-embed: status=%d body=%s", resp.StatusCode, string(raw))
	}
	var out openAIEmbedResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("openai-embed: decode: %w", err)
	}
	if out.Error != nil {
		return nil, fmt.Errorf("openai-embed: %s: %s", out.Error.Type, out.Error.Message)
	}
	vecs := make([][]float32, len(out.Data))
	for i, d := range out.Data {
		vecs[i] = d.Embedding
	}
	return vecs, nil
}
