package main

// MockProvider is a deterministic Provider used by tests + offline CI.
// It returns text that quotes the leading 200 chars of the request's System
// prompt, so a test can assert "the right context was passed in".

import (
	"context"
	"strings"
)

// MockProvider satisfies Provider without any network call.
type MockProvider struct {
	// Calls records every Complete invocation in order — tests can inspect it.
	Calls []CompleteRequest
}

// NewMockProvider builds an empty mock.
func NewMockProvider() *MockProvider { return &MockProvider{} }

// Complete echoes a deterministic response shaped like a real LLM answer.
// The shape is: a one-line preamble + the first 200 chars of the System
// prompt's "Context:" section. This lets TestPipelineEndToEndWithMock assert
// that the retrieved chunks made it into the context window.
func (m *MockProvider) Complete(ctx context.Context, req CompleteRequest) (CompleteResponse, error) {
	m.Calls = append(m.Calls, req)
	// Mimic deterministic LLM output keyed only on the System prompt — that
	// way a test can call Complete twice and get byte-identical results.
	preamble := "[mock-llm] answer based on retrieved context:\n"
	body := req.System
	// Snip out everything before the "Context:" anchor that pipeline.go
	// inserts; this is the load-bearing slice for assertions.
	if idx := strings.Index(body, "Context:"); idx >= 0 {
		body = body[idx:]
	}
	// Cap the echoed slice at 200 runes so the response stays tidy.
	if rs := []rune(body); len(rs) > 200 {
		body = string(rs[:200])
	}
	return CompleteResponse{
		Text:         preamble + body,
		InputTokens:  len(req.System) + len(req.Messages)*8,
		OutputTokens: len(preamble) + len(body),
	}, nil
}
