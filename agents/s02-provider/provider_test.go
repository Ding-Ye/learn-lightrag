package main

// Tests for s02 cover all three Provider impls. CI runs these in the
// matrix — none touch the real OpenAI; the OpenAIProvider tests use
// httptest.NewServer to fake the upstream response.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// TestProviderInterfaceContract is a compile-time guard ensuring all three
// concrete types satisfy Provider. If anyone removes a method, this fails
// to build.
func TestProviderInterfaceContract(t *testing.T) {
	var _ Provider = (*OpenAIProvider)(nil)
	var _ Provider = (*MockProvider)(nil)
	var _ Provider = (*EchoProvider)(nil)
}

// TestOpenAIProviderHappyPath spins up an httptest server that responds
// like OpenAI's chat-completions API. We then assert (a) the request body
// shape (model + messages + system are correctly assembled) and (b) the
// response decodes into Text + token counts.
func TestOpenAIProviderHappyPath(t *testing.T) {
	var gotPayload openAIChatPayload
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("auth header = %q, want Bearer test-key", got)
		}
		if err := json.NewDecoder(r.Body).Decode(&gotPayload); err != nil {
			t.Fatalf("decode payload: %v", err)
		}
		_, _ = w.Write([]byte(`{
			"choices": [{"message": {"role": "assistant", "content": "Paris."}}],
			"usage":   {"prompt_tokens": 12, "completion_tokens": 1}
		}`))
	}))
	defer server.Close()

	p := NewOpenAIProvider(
		WithAPIKey("test-key"),
		WithBaseURL(server.URL),
		WithModel("gpt-test"),
	)
	resp, err := p.Complete(context.Background(), CompleteRequest{
		System:   "You are concise.",
		Messages: []Message{{Role: "user", Content: "What is the capital of France?"}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.Text != "Paris." {
		t.Errorf("Text = %q, want %q", resp.Text, "Paris.")
	}
	if resp.InputTokens != 12 || resp.OutputTokens != 1 {
		t.Errorf("token counts = (%d,%d), want (12,1)", resp.InputTokens, resp.OutputTokens)
	}

	// Verify the request body shape.
	if gotPayload.Model != "gpt-test" {
		t.Errorf("payload model = %q, want gpt-test", gotPayload.Model)
	}
	if len(gotPayload.Messages) != 2 {
		t.Fatalf("payload messages len = %d, want 2 (system + user)", len(gotPayload.Messages))
	}
	if gotPayload.Messages[0].Role != "system" || gotPayload.Messages[0].Content != "You are concise." {
		t.Errorf("message[0] = %+v, want system/concise", gotPayload.Messages[0])
	}
	if gotPayload.Messages[1].Role != "user" || !strings.Contains(gotPayload.Messages[1].Content, "France") {
		t.Errorf("message[1] = %+v, want user/France", gotPayload.Messages[1])
	}
	if gotPayload.Stream {
		t.Errorf("stream = true, expected false in s02")
	}
}

// TestOpenAIProviderRetriesOn429 returns 429 twice then 200. We assert the
// retry loop made exactly 3 attempts and ultimately succeeded — i.e. the
// transient-status detection works AND the backoff did not spuriously give up.
func TestOpenAIProviderRetriesOn429(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := calls.Add(1)
		if n <= 2 {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"error":{"message":"slow down","type":"rate_limit"}}`))
			return
		}
		_, _ = w.Write([]byte(`{
			"choices": [{"message": {"role": "assistant", "content": "ok"}}],
			"usage":   {"prompt_tokens": 1, "completion_tokens": 1}
		}`))
	}))
	defer server.Close()

	p := NewOpenAIProvider(
		WithAPIKey("k"),
		WithBaseURL(server.URL),
		WithMaxRetries(2), // 1 initial + 2 retries = 3 total attempts
	)
	resp, err := p.Complete(context.Background(), CompleteRequest{
		Messages: []Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Complete after retries: %v", err)
	}
	if resp.Text != "ok" {
		t.Errorf("Text = %q, want ok", resp.Text)
	}
	if got := calls.Load(); got != 3 {
		t.Errorf("server saw %d calls, want 3 (1 + 2 retries)", got)
	}
}

// TestOpenAIProviderRespectsContextCancel cancels the context mid-request.
// The handler sleeps 500ms; we cancel after 50ms; Complete must return
// promptly and the error must wrap context.Canceled.
func TestOpenAIProviderRespectsContextCancel(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-time.After(500 * time.Millisecond):
			_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"too late"}}]}`))
		case <-r.Context().Done():
			return
		}
	}))
	defer server.Close()

	p := NewOpenAIProvider(
		WithAPIKey("k"),
		WithBaseURL(server.URL),
		WithMaxRetries(0),
	)
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()
	start := time.Now()
	_, err := p.Complete(ctx, CompleteRequest{
		Messages: []Message{{Role: "user", Content: "hi"}},
	})
	if err == nil {
		t.Fatalf("expected context cancel error")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want errors.Is(err, context.Canceled)", err)
	}
	if elapsed := time.Since(start); elapsed > 300*time.Millisecond {
		t.Errorf("Complete took %v, expected to return quickly after cancel", elapsed)
	}
}

// TestMockProviderDeterminism asserts that calling Complete twice with the
// SAME CompleteRequest yields byte-identical responses. This property is
// what makes MockProvider safe in CI.
func TestMockProviderDeterminism(t *testing.T) {
	m := NewMockProvider(nil)
	req := CompleteRequest{
		Model:    "gpt-4o-mini",
		System:   "you are a test",
		Messages: []Message{{Role: "user", Content: "hi"}},
	}
	a, err := m.Complete(context.Background(), req)
	if err != nil {
		t.Fatalf("first Complete: %v", err)
	}
	b, err := m.Complete(context.Background(), req)
	if err != nil {
		t.Fatalf("second Complete: %v", err)
	}
	if a.Text != b.Text {
		t.Errorf("non-deterministic: %q vs %q", a.Text, b.Text)
	}
	if got := m.Count(); got != 2 {
		t.Errorf("Count() = %d, want 2", got)
	}
	if len(m.Calls) != 2 {
		t.Errorf("len(Calls) = %d, want 2", len(m.Calls))
	}
}

// TestEchoProviderReturnsSystemPrompt is the trivial smoke test that
// motivates EchoProvider's existence: the system prompt round-trips
// verbatim into Text.
func TestEchoProviderReturnsSystemPrompt(t *testing.T) {
	e := NewEchoProvider()
	system := "this is the system prompt — please echo me back."
	resp, err := e.Complete(context.Background(), CompleteRequest{
		System:   system,
		Messages: []Message{{Role: "user", Content: "irrelevant"}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.Text != system {
		t.Errorf("Text = %q, want %q", resp.Text, system)
	}
}

// TestOpenAIProviderUnauthorizedNotRetried asserts that 401 is treated as
// permanent: no retries, no exhaustion message — just the auth error.
// This is the load-bearing distinction between transient and permanent.
func TestOpenAIProviderUnauthorizedNotRetried(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer server.Close()

	p := NewOpenAIProvider(
		WithAPIKey("bad"),
		WithBaseURL(server.URL),
		WithMaxRetries(3),
	)
	_, err := p.Complete(context.Background(), CompleteRequest{
		Messages: []Message{{Role: "user", Content: "hi"}},
	})
	if err == nil {
		t.Fatalf("expected error from 401")
	}
	if !strings.Contains(err.Error(), "401") {
		t.Errorf("err = %v, want it to mention 401", err)
	}
	if got := calls.Load(); got != 1 {
		t.Errorf("server saw %d calls, want 1 (401 must not retry)", got)
	}
}

// helper to silence vet's unused fmt warning when developing tests.
var _ = fmt.Sprintf
