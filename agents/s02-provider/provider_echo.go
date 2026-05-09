package main

// EchoProvider returns the request's System prompt verbatim as the
// response Text. It exists for trivial smoke tests and dry-runs (e.g.
// "did the caller assemble the right system prompt?"), and for
// pedagogical purposes — it's the smallest possible Provider impl.

import "context"

// EchoProvider satisfies Provider with zero state.
type EchoProvider struct{}

// NewEchoProvider builds an EchoProvider. Stateless, so always the same
// instance is also fine — but a constructor keeps the API consistent
// with the other two providers.
func NewEchoProvider() *EchoProvider { return &EchoProvider{} }

// Complete returns the system prompt verbatim. Token counts are sized
// by string length over 4 (approximate), matching MockProvider.
func (e *EchoProvider) Complete(ctx context.Context, req CompleteRequest) (CompleteResponse, error) {
	if err := ctx.Err(); err != nil {
		return CompleteResponse{}, err
	}
	return CompleteResponse{
		Text:         req.System,
		InputTokens:  len(req.System) / 4,
		OutputTokens: len(req.System) / 4,
	}, nil
}
