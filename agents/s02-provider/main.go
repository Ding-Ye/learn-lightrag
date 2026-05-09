package main

// CLI for s02: demo the same Provider interface with three different impls.
// Run examples:
//   go run . -provider mock  -q "What's the capital of France?"
//   go run . -provider echo  -q "anything"
//   go run . -provider openai -model gpt-4o-mini -q "What's the capital of France?"
//
// With -v, the call also prints input/output tokens after the call so the
// learner can see the s02-vs-s01 delta (token accounting now actually
// populates CompleteResponse fields).

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"time"
)

func main() {
	provName := flag.String("provider", "mock", "provider: openai | mock | echo")
	model := flag.String("model", "gpt-4o-mini", "model name (only used by openai)")
	question := flag.String("q", "Hello!", "question to ask")
	verbose := flag.Bool("v", false, "verbose: print token accounting after the call")
	timeout := flag.Duration("timeout", 45*time.Second, "per-request timeout (only openai)")
	flag.Parse()

	prov, err := buildProvider(*provName, *model, *timeout, *verbose)
	if err != nil {
		log.Fatalf("s02: build provider: %v", err)
	}

	system := "You are a concise assistant. Answer in one sentence."
	req := CompleteRequest{
		Model:       *model,
		System:      system,
		Messages:    []Message{{Role: "user", Content: *question}},
		MaxTokens:   200,
		Temperature: 0.0,
	}

	ctx, cancel := context.WithTimeout(context.Background(), *timeout+5*time.Second)
	defer cancel()
	resp, err := prov.Complete(ctx, req)
	if err != nil {
		log.Fatalf("s02: complete: %v", err)
	}

	fmt.Println("=== Answer ===")
	fmt.Println(resp.Text)
	if *verbose {
		fmt.Println()
		fmt.Println("=== Tokens ===")
		fmt.Printf("input=%d output=%d  (s02 finally populates these — s01 left them best-effort)\n",
			resp.InputTokens, resp.OutputTokens)
	}
}

// buildProvider resolves the -provider flag. Adding a Phase-G provider
// (Anthropic / Bedrock / Ollama) means one new case here — the rest of
// the program never has to change.
func buildProvider(name, model string, timeout time.Duration, verbose bool) (Provider, error) {
	switch name {
	case "openai":
		var logger Logger = noopLogger
		if verbose {
			logger = func(format string, args ...any) {
				fmt.Fprintf(os.Stderr, "[openai] "+format+"\n", args...)
			}
		}
		return NewOpenAIProvider(
			WithModel(model),
			WithTimeout(timeout),
			WithMaxRetries(2),
			WithLogger(logger),
		), nil
	case "mock":
		return NewMockProvider(nil), nil
	case "echo":
		return NewEchoProvider(), nil
	default:
		return nil, fmt.Errorf("unknown provider %q (want openai | mock | echo)", name)
	}
}
