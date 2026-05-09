package main

// MockProvider is a deterministic, no-network Provider for tests + offline
// CI. Compared to s01, s02's mock is upgraded with:
//   - sha256-based default response (so identical CompleteRequest values
//     yield byte-identical responses across runs / processes)
//   - a swappable Resp(req) callback so tests can inject custom fixtures
//   - an atomic counter Count() so tests assert call shape without
//     iterating Calls

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sync/atomic"
)

// MockProvider satisfies Provider without any network call.
type MockProvider struct {
	// Calls records every Complete invocation in order.
	Calls []CompleteRequest

	// Resp lets tests swap the response shape. If nil, defaults to a
	// hash-of-request canned response — useful for snapshot-style tests.
	Resp func(CompleteRequest) string

	// counter is incremented on each Complete. Atomic so concurrent
	// goroutine callers can read it without external locking.
	counter atomic.Int64
}

// NewMockProvider builds an empty mock. Pass nil resp for deterministic
// hash-echo behavior, or a closure for custom canned answers.
func NewMockProvider(resp func(CompleteRequest) string) *MockProvider {
	return &MockProvider{Resp: resp}
}

// Count returns how many Complete calls have happened. Useful in tests
// to assert e.g. retry behavior didn't accidentally invoke the upstream
// extra times.
func (m *MockProvider) Count() int64 { return m.counter.Load() }

// Complete records the call, increments the counter, and returns the
// result of Resp (or the default hash echo).
func (m *MockProvider) Complete(ctx context.Context, req CompleteRequest) (CompleteResponse, error) {
	if err := ctx.Err(); err != nil {
		return CompleteResponse{}, err
	}
	m.Calls = append(m.Calls, req)
	m.counter.Add(1)
	var text string
	if m.Resp != nil {
		text = m.Resp(req)
	} else {
		text = defaultMockResponse(req)
	}
	// Token counts: rough approximation by content size. Real LLMs do
	// proper BPE; tests don't need accuracy, only determinism.
	in := len(req.System)
	for _, msg := range req.Messages {
		in += len(msg.Content)
	}
	return CompleteResponse{
		Text:         text,
		InputTokens:  in / 4,
		OutputTokens: len(text) / 4,
	}, nil
}

// defaultMockResponse hashes the entire CompleteRequest so two identical
// requests yield identical answers. This is the load-bearing property
// for test determinism — see TestMockProviderDeterminism.
func defaultMockResponse(req CompleteRequest) string {
	// JSON-encode the request to feed the hasher. Field order of struct
	// fields is stable in Go, so the encoded bytes are deterministic.
	b, _ := json.Marshal(req)
	sum := sha256.Sum256(b)
	return "[mock-llm] sha256=" + hex.EncodeToString(sum[:8]) + " (deterministic echo)"
}
