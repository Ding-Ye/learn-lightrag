package main

// CLI for s01: load `testdata/book.txt`, insert it, ask a question.
// Mirrors examples/lightrag_openai_demo.py end-to-end shape.

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
)

func main() {
	q := flag.String("q", "", "question to ask after ingesting the doc")
	docPath := flag.String("doc", "testdata/book.txt", "path to text doc to ingest")
	provFlag := flag.String("provider", "", "provider: openai | mock (default: env-derived)")
	verbose := flag.Bool("v", false, "verbose: print retrieval debug lines")
	flag.Parse()

	provName := pickProvider(*provFlag)
	prov, embedder, err := buildProviders(provName)
	if err != nil {
		log.Fatalf("s01: build providers: %v", err)
	}

	pipe := NewPipeline(prov, embedder, NewInMemoryVectorStore(), NewMemoryKVStore())
	pipe.Verbose = *verbose
	pipe.Logf = func(format string, a ...any) { fmt.Fprintf(os.Stderr, format+"\n", a...) }

	text, err := os.ReadFile(*docPath)
	if err != nil {
		log.Fatalf("s01: read doc %q: %v", *docPath, err)
	}

	ctx := context.Background()
	docID := "book"
	if err := pipe.Insert(ctx, docID, string(text)); err != nil {
		log.Fatalf("s01: insert: %v", err)
	}
	fmt.Fprintf(os.Stderr, "[s01] ingested doc %q (provider=%s)\n", *docPath, provName)

	if *q == "" {
		fmt.Fprintln(os.Stderr, "[s01] no -q given; ingestion-only run, exiting.")
		return
	}
	res, err := pipe.Query(ctx, *q, 3)
	if err != nil {
		log.Fatalf("s01: query: %v", err)
	}
	fmt.Println("=== Answer ===")
	fmt.Println(res.Content)
	fmt.Println()
	fmt.Println("=== References ===")
	for _, ref := range res.References {
		fmt.Println("- " + ref)
	}
	fmt.Printf("\n[mode=%s]\n", res.Mode)
}

// pickProvider resolves the -provider flag: explicit value wins; otherwise
// "openai" if OPENAI_API_KEY is set, else "mock" so CI can run dry.
func pickProvider(flagVal string) string {
	if flagVal != "" {
		return flagVal
	}
	if os.Getenv("OPENAI_API_KEY") != "" {
		return "openai"
	}
	return "mock"
}

// buildProviders constructs a (Provider, EmbeddingProvider) pair based on
// the resolved name. Adding a new provider in Phase G means one new case here.
func buildProviders(name string) (Provider, EmbeddingProvider, error) {
	switch name {
	case "openai":
		return NewOpenAIProvider(), NewOpenAIEmbedder(), nil
	case "mock":
		return NewMockProvider(), NewMockEmbedder(), nil
	default:
		return nil, nil, fmt.Errorf("unknown provider %q (want openai | mock)", name)
	}
}
