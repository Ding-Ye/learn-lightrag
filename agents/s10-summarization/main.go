package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"strings"
)

// main.go is the s10 CLI demo.  Constructs 12 fake descriptions for one
// entity (Eleanor Hartwell, the recurring character from earlier sessions),
// runs SummarizeDescriptions, and prints the BEFORE (raw concat) vs AFTER
// (summarized) comparison alongside the llmUsed flag and token-count diff.
//
// Usage:
//   go run . [-provider mock]
//
// Default: mock provider — offline, deterministic, no network.  The mock
// provider is the only path s10 ships; -provider openai is intentionally
// unwired here (see s02 for the real OpenAI implementation).

func main() {
	provName := flag.String("provider", "mock", "LLM provider: mock (no network) or openai (unwired in s10)")
	flag.Parse()

	if err := run(context.Background(), *provName); err != nil {
		fmt.Fprintf(os.Stderr, "s10 demo failed: %v\n", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, provName string) error {
	prov, err := buildProvider(provName)
	if err != nil {
		return err
	}

	descriptions := eleanorDescriptions()

	rawConcat := strings.Join(descriptions, "\n\n")

	fmt.Println("=== s10 demo: Eleanor Hartwell appears in 12 chunks ===")
	fmt.Printf("\n[a] raw concatenation (what s09 would have stored):\n")
	fmt.Println(strings.Repeat("-", 60))
	fmt.Println(rawConcat)
	fmt.Printf("\nraw token count (whitespace-word stub): %d\n", tokenCount(rawConcat))
	fmt.Printf("description count: %d\n", len(descriptions))

	// Tight budget so the LLM path triggers (12 descriptions exceed
	// countThreshold=6 anyway, but we lower budget to make recursion likely).
	summary, llmUsed, err := SummarizeDescriptions(ctx, prov, descriptions,
		120 /*budgetTokens*/, 80 /*contextSize*/, 6 /*countThreshold*/)
	if err != nil {
		return fmt.Errorf("SummarizeDescriptions: %w", err)
	}

	fmt.Printf("\n[b] s10 summarized version:\n")
	fmt.Println(strings.Repeat("-", 60))
	fmt.Println(summary)
	fmt.Printf("\nsummary token count: %d\n", tokenCount(summary))
	fmt.Printf("[c] llmUsed flag: %v\n", llmUsed)

	rawTokens := tokenCount(rawConcat)
	sumTokens := tokenCount(summary)
	pct := 100
	if rawTokens > 0 {
		pct = 100 * sumTokens / rawTokens
	}
	fmt.Printf("\n[d] compression: %d → %d tokens (%d%% of original)\n",
		rawTokens, sumTokens, pct)

	// Bonus: also exercise the "small list" fast-path (no LLM).
	fmt.Printf("\n=== fast-path demo: 3 descriptions, plenty of budget ===\n")
	short := descriptions[:3]
	fastSummary, fastLLM, err := SummarizeDescriptions(ctx, prov, short, 500, 2000, 6)
	if err != nil {
		return fmt.Errorf("fast-path: %w", err)
	}
	fmt.Printf("llmUsed: %v (expect false — under threshold AND under budget)\n", fastLLM)
	fmt.Printf("output is exact concat? %v\n", fastSummary == strings.Join(short, "\n\n"))

	return nil
}

// eleanorDescriptions returns 12 fake description fragments for the
// recurring character "Eleanor Hartwell" (appearing across earlier
// sessions' demos).  Hand-written to look like what s09's extractor
// would produce: short, third-person, varied focus.
func eleanorDescriptions() []string {
	return []string{
		"Eleanor Hartwell is a senior archivist at the Whitehall Museum, known for her meticulous restoration work.",
		"Eleanor Hartwell discovered a lost manuscript believed to date from the 12th century during a routine cataloguing project.",
		"Eleanor Hartwell collaborated with Cambridge historians to verify the provenance of the Hartwell Codex.",
		"Eleanor Hartwell published a paper in the Journal of Medieval Studies arguing that the codex predated the Magna Carta.",
		"Eleanor Hartwell was awarded the Royal Society's Whitechapel Medal for her contributions to historical conservation.",
		"Eleanor Hartwell taught a graduate seminar at Oxford on manuscript authentication techniques.",
		"Eleanor Hartwell mentored younger archivists, several of whom now lead departments at major European institutions.",
		"Eleanor Hartwell was a key witness during a parliamentary inquiry into the cultural-property repatriation debate.",
		"Eleanor Hartwell consulted on a BBC documentary about lost medieval libraries, attracting wider public attention.",
		"Eleanor Hartwell suffered a brief professional setback when an early carbon-dating result was challenged on review.",
		"Eleanor Hartwell, after the disputed result was overturned, returned to her position with the museum's full backing.",
		"Eleanor Hartwell continues to advocate for open access to digitized medieval manuscripts in the British Library.",
	}
}

// --- Provider selection -------------------------------------------------------

func buildProvider(name string) (Provider, error) {
	switch name {
	case "mock", "":
		return NewMockProvider(), nil
	case "openai":
		// s02 owns the real OpenAI implementation; s10 stays offline so
		// `go test` and `go run .` never need a network connection.
		return nil, errors.New("openai provider not wired in s10 (use s02 or run with -provider mock)")
	default:
		return nil, fmt.Errorf("unknown provider: %s", name)
	}
}

// --- MockProvider -------------------------------------------------------------

// MockProvider is the offline-deterministic Provider implementation used by
// every test in this session and the default `go run .` flow.  It returns a
// canned summary that's intentionally SHORTER than the input so the demo
// shows real compression and the budget tests can verify shrinkage.
type MockProvider struct {
	// Calls counts how many times Complete was invoked.  Plain int — tests
	// drive the summarizer sequentially.
	Calls int

	// FixedResponse, when non-empty, overrides the canned response.  Used
	// by tests to inject a specific summary length.
	FixedResponse string

	// SleepMs lets cancellation tests force a delay.
	SleepMs int
}

// NewMockProvider returns a fresh MockProvider.
func NewMockProvider() *MockProvider {
	return &MockProvider{}
}

// Complete implements Provider.  Returns either FixedResponse or a synthesized
// "summary" derived by trimming the JSONL block to a few sentences.
func (m *MockProvider) Complete(ctx context.Context, req CompleteRequest) (CompleteResponse, error) {
	m.Calls++
	if m.SleepMs > 0 {
		// Honor cancellation while sleeping.
		select {
		case <-ctx.Done():
			return CompleteResponse{}, ctx.Err()
		default:
		}
	}
	if err := ctx.Err(); err != nil {
		return CompleteResponse{}, err
	}
	if m.FixedResponse != "" {
		return CompleteResponse{Text: m.FixedResponse, InputTokens: 100, OutputTokens: 30}, nil
	}
	// Synthesized response: ~12 words, deterministic.  This is the canned
	// summary used by the CLI demo and the basic LLM tests.
	canned := "The merged subject, drawn from multiple chunks, is a recurring entity characterized by their professional accomplishments and stable reputation in their field."
	return CompleteResponse{Text: canned, InputTokens: 200, OutputTokens: 24}, nil
}

// --- compile-time interface assertions ---------------------------------------

var _ Provider = (*MockProvider)(nil)
