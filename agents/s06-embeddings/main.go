package main

// CLI for s06: embed every line of testdata/sample.txt and report
// throughput. The two providers ("openai" + "mock") share the same
// EmbeddingProvider contract; switching providers is a flag.
//
// Run examples:
//   go run . -provider mock                                # offline default
//   go run . -provider mock -batch 16                      # smaller slicing
//   go run . -provider openai -model text-embedding-3-small  # needs OPENAI_API_KEY

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"
)

func main() {
	provName := flag.String("provider", "mock", "provider: openai | mock")
	model := flag.String("model", "text-embedding-3-small", "embedding model (only used by openai)")
	batch := flag.Int("batch", 128, "batch size per HTTP call (only matters for openai)")
	dataPath := flag.String("data", "testdata/sample.txt", "newline-separated input file")
	manifest := flag.Bool("manifest", true, "print JSON manifest of (text, vec_first_6_dims) to stdout")
	flag.Parse()

	texts, err := readLines(*dataPath)
	if err != nil {
		log.Fatalf("s06: read %s: %v", *dataPath, err)
	}

	embedder, providerLabel, err := buildEmbedder(*provName, *model, *batch)
	if err != nil {
		log.Fatalf("s06: build embedder: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	start := time.Now()
	vecs, err := embedder.Embed(ctx, texts)
	if err != nil {
		log.Fatalf("s06: embed: %v", err)
	}
	elapsed := time.Since(start)

	throughput := float64(len(texts)) / elapsed.Seconds()
	fmt.Println("=== s06 embedder demo ===")
	fmt.Printf("provider:   %s\n", providerLabel)
	fmt.Printf("model:      %s\n", *model)
	fmt.Printf("batch:      %d\n", *batch)
	fmt.Printf("texts:      %d\n", len(texts))
	fmt.Printf("dim:        %d\n", embedder.Dim())
	fmt.Printf("elapsed:    %s\n", elapsed)
	fmt.Printf("throughput: %.1f lines/sec\n", throughput)

	if !*manifest {
		return
	}
	fmt.Println()
	fmt.Println("=== manifest (first 6 dims of each vector) ===")
	type entry struct {
		Text     string    `json:"text"`
		Preview  []float32 `json:"vector_first_6_dims"`
		Dim      int       `json:"dim"`
	}
	out := make([]entry, 0, len(vecs))
	for i, v := range vecs {
		preview := v
		if len(v) > 6 {
			preview = v[:6]
		}
		out = append(out, entry{
			Text:    texts[i],
			Preview: preview,
			Dim:     len(v),
		})
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(out); err != nil {
		log.Fatalf("s06: encode manifest: %v", err)
	}
}

// buildEmbedder resolves the -provider flag. Adding a Phase-G provider
// means one new case here — the rest of the program never has to change.
func buildEmbedder(name, model string, batch int) (EmbeddingProvider, string, error) {
	switch name {
	case "openai":
		return NewOpenAIEmbedder(
			WithModel(model),
			WithBatchSize(batch),
			WithMaxRetries(3),
		), "openai", nil
	case "mock":
		return NewMockEmbedder(), "mock", nil
	default:
		return nil, "", fmt.Errorf("unknown provider %q (want openai | mock)", name)
	}
}

// readLines reads a newline-separated text file and returns one entry per
// non-empty line (trimmed). Single source of truth for the demo's input.
func readLines(path string) ([]string, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	out := make([]string, 0, 64)
	for _, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		out = append(out, line)
	}
	return out, nil
}
