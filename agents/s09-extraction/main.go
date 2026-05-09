package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"
)

// main.go is the s09 CLI demo.  Reads a doc, splits to ~3 paragraph chunks,
// runs Extractor.Extract on each, prints entities + relations + cache hit
// ratio.
//
// Usage:
//   go run . [-provider mock|openai] [-rounds N] [-doc PATH] [-runs N]
//
// Default: mock provider (offline, deterministic, what CI uses).
// `-rounds 2` exercises gleaning past the upstream default of 1.
// Re-run with the same args to see cache hit ratio jump to 100%.

func main() {
	provName := flag.String("provider", "mock", "LLM provider: mock or openai")
	rounds := flag.Int("rounds", 1, "max gleaning rounds after initial extraction")
	docPath := flag.String("doc", "./testdata/sample.txt", "path to text doc")
	runs := flag.Int("runs", 1, "how many times to extract (>1 demonstrates cache hits)")
	flag.Parse()

	ctx := context.Background()
	if err := run(ctx, *provName, *rounds, *docPath, *runs); err != nil {
		fmt.Fprintf(os.Stderr, "s09 demo failed: %v\n", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, provName string, rounds int, docPath string, runs int) error {
	// Read the doc.  We bake a fallback sample inline so `go run .` works
	// out of the box even before the user creates testdata/sample.txt.
	docText, err := loadDocOrFallback(docPath)
	if err != nil {
		return fmt.Errorf("load doc: %w", err)
	}

	chunks := paragraphChunks(docText, "demo-doc")
	if len(chunks) == 0 {
		return errors.New("no chunks parsed from doc")
	}
	fmt.Printf("loaded %d chunks from %s\n", len(chunks), docPath)

	prov, err := buildProvider(provName)
	if err != nil {
		return fmt.Errorf("build provider: %w", err)
	}

	cache := NewMemoryKVStore()
	extractor := NewExtractor(prov,
		WithModel("gpt-4o-mini"),
		WithGleaningRounds(rounds),
		WithCache(cache),
		WithLogger(stdLogger{l: log.New(os.Stderr, "[extractor] ", 0)}),
	)

	for runIdx := 0; runIdx < runs; runIdx++ {
		if runs > 1 {
			fmt.Printf("\n==== run %d/%d ====\n", runIdx+1, runs)
		}
		runOnce(ctx, extractor, chunks)
	}

	stats := extractor.Stats()
	total := stats.CacheHits + stats.CacheMisses
	pct := 0.0
	if total > 0 {
		pct = 100.0 * float64(stats.CacheHits) / float64(total)
	}
	fmt.Printf("\nfinal: cache hits %d / %d (%.0f%%) | LLM calls: %d\n",
		stats.CacheHits, total, pct, stats.LLMCalls)

	return nil
}

func runOnce(ctx context.Context, extractor *Extractor, chunks []Chunk) {
	for i, ch := range chunks {
		ents, rels, err := extractor.Extract(ctx, ch)
		if err != nil {
			fmt.Printf("chunk %d: extract failed: %v\n", i+1, err)
			continue
		}
		fmt.Printf("\n=== chunk %d (%s): %d entities, %d relations ===\n",
			i+1, ch.ChunkID(), len(ents), len(rels))
		for _, e := range ents {
			fmt.Printf("  entity   %-15s type=%-10s %q\n",
				e.Name, e.Type, truncStr(e.Description, 50))
		}
		for _, r := range rels {
			fmt.Printf("  relation %-12s -- %-12s keywords=%-25q %q\n",
				r.SrcID, r.TgtID, r.Keywords, truncStr(r.Description, 40))
		}
	}
}

func truncStr(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + "…"
}

// paragraphChunks splits text into ~paragraph-sized chunks (separated by
// blank lines).  This is intentionally a stub — s04 has the real
// token-based chunker.  Empty and whitespace-only paragraphs are skipped.
func paragraphChunks(text string, docID string) []Chunk {
	parts := strings.Split(text, "\n\n")
	chunks := make([]Chunk, 0, len(parts))
	idx := 0
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		chunks = append(chunks, Chunk{
			ContentDocID:    docID,
			Content:         p,
			Tokens:          len(strings.Fields(p)), // word count stub
			ChunkOrderIndex: idx,
		})
		idx++
	}
	return chunks
}

func loadDocOrFallback(path string) (string, error) {
	if path != "" {
		if b, err := os.ReadFile(path); err == nil {
			return string(b), nil
		}
	}
	// Inline fallback so `go run .` works without setup.
	return strings.Join([]string{
		"Scrooge was a miserly old businessman who hated Christmas. " +
			"His former business partner, Marley, had died seven years before.",
		"On Christmas Eve, the ghost of Jacob Marley visited Scrooge in his counting-house. " +
			"Marley warned Scrooge that three more spirits would visit before dawn.",
		"The Ghost of Christmas Past took Scrooge to revisit his lonely childhood. " +
			"He saw himself as a young clerk under Mr. Fezziwig.",
	}, "\n\n"), nil
}

// --- Provider selection ---

func buildProvider(name string) (Provider, error) {
	switch name {
	case "mock", "":
		return NewMockProvider(), nil
	case "openai":
		// We deliberately do NOT implement OpenAI here — s02 owns that.
		// s09 sticks to the mock provider for offline determinism.
		return nil, errors.New("openai provider not wired in s09 (use s02 or run with -provider mock)")
	default:
		return nil, fmt.Errorf("unknown provider: %s", name)
	}
}

// --- MockProvider ---

// MockProvider is the offline-deterministic Provider implementation used
// by every test in this session and the default `go run .` flow.  It
// keys responses by sha256(system+lastUserContent) so the same input
// always returns the same output — important for cache-hit testing.
//
// The provider has two response modes:
//
//  1. continue: if the LAST user message is the gleaning continuation
//     prompt, return the "round 2" response (one new entity per chunk).
//  2. initial: otherwise, return the "round 1" response derived from the
//     chunk content.  The keying logic recognizes a few canned chunk
//     prefixes and otherwise falls back to a generic synthetic response.
//
// Sleeps are inserted between header and body when SleepMs > 0 — used by
// TestExtractionRespectsContextCancel.
type MockProvider struct {
	SleepMs int

	// CallCount lets tests assert how many times the provider was hit.
	// Plain int — not goroutine-safe; tests drive the extractor
	// sequentially.
	CallCount int

	// ResponseOverride lets tests inject a specific response keyed by
	// the LAST user message content (or a substring of it).  Lookup is
	// substring-match; first match wins.  Set this BEFORE calling Extract
	// to override the canned responses.
	ResponseOverride map[string]string
}

// NewMockProvider returns a fresh MockProvider with no sleeps and no
// overrides.
func NewMockProvider() *MockProvider {
	return &MockProvider{ResponseOverride: make(map[string]string)}
}

// Complete implements Provider.
func (m *MockProvider) Complete(ctx context.Context, req CompleteRequest) (CompleteResponse, error) {
	m.CallCount++

	if m.SleepMs > 0 {
		select {
		case <-time.After(time.Duration(m.SleepMs) * time.Millisecond):
		case <-ctx.Done():
			return CompleteResponse{}, ctx.Err()
		}
	}
	if err := ctx.Err(); err != nil {
		return CompleteResponse{}, err
	}

	// Find the last user message — the one we route on for round detection.
	lastUser := ""
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if req.Messages[i].Role == "user" {
			lastUser = req.Messages[i].Content
			break
		}
	}

	// Determine round: if the LAST user message is a continuation prompt,
	// we're in a gleaning round.  In that case the chunk content lives in
	// the FIRST user message (the initial extraction prompt).
	isContinue := strings.Contains(lastUser, "MISSED or") ||
		strings.Contains(lastUser, "Based on the previous extraction")

	// For routing/override matching we want the original chunk text.
	// On continue rounds it's the FIRST user message; on initial it's the
	// last (which is the only user message anyway).
	routeMsg := lastUser
	if isContinue {
		for _, msg := range req.Messages {
			if msg.Role == "user" {
				routeMsg = msg.Content
				break
			}
		}
	}

	// Override path: substring match against ResponseOverride.
	for needle, resp := range m.ResponseOverride {
		if strings.Contains(routeMsg, needle) {
			return CompleteResponse{Text: resp, InputTokens: 100, OutputTokens: 200}, nil
		}
	}

	text := mockCannedResponse(routeMsg, isContinue)
	return CompleteResponse{Text: text, InputTokens: 80, OutputTokens: 120}, nil
}

// mockCannedResponse picks a canned delimited extraction response based
// on the chunk content, so the demo produces meaningful output without
// touching a real LLM.  Fall-through is a generic 1-entity response so
// any unrecognized chunk still parses to something.
func mockCannedResponse(userPrompt string, isContinue bool) string {
	// Heuristic: pull out the last input chunk text from the user prompt
	// (between ``` fences if present).
	chunk := userPrompt
	if i := strings.Index(userPrompt, "```"); i >= 0 {
		rest := userPrompt[i+3:]
		if j := strings.Index(rest, "```"); j >= 0 {
			chunk = strings.TrimSpace(rest[:j])
		}
	}
	chunkLower := strings.ToLower(chunk)
	hash := keyHash(chunk)

	switch {
	case strings.Contains(chunkLower, "scrooge") && strings.Contains(chunkLower, "marley"):
		if isContinue {
			return "" +
				"entity" + tupleDelimiter + "Christmas Eve" + tupleDelimiter + "event" + tupleDelimiter + "the night before Christmas when the spirits visit Scrooge.\n" +
				completionDelimiter + "\n"
		}
		return "" +
			"entity" + tupleDelimiter + "Scrooge" + tupleDelimiter + "person" + tupleDelimiter + "a miserly old businessman who hates Christmas.\n" +
			"entity" + tupleDelimiter + "Marley" + tupleDelimiter + "person" + tupleDelimiter + "Scrooge's deceased former business partner.\n" +
			"relation" + tupleDelimiter + "Scrooge" + tupleDelimiter + "Marley" + tupleDelimiter + "partnership" + tupleDelimiter + "Scrooge and Marley were business partners for many years.\n" +
			completionDelimiter + "\n"

	case strings.Contains(chunkLower, "fezziwig") || strings.Contains(chunkLower, "ghost of christmas past"):
		if isContinue {
			return "" +
				"entity" + tupleDelimiter + "Mr. Fezziwig" + tupleDelimiter + "person" + tupleDelimiter + "Scrooge's apprentice-master, a kind employer.\n" +
				completionDelimiter + "\n"
		}
		return "" +
			"entity" + tupleDelimiter + "Ghost of Christmas Past" + tupleDelimiter + "concept" + tupleDelimiter + "the first spirit to visit Scrooge, showing him his childhood.\n" +
			"entity" + tupleDelimiter + "Scrooge" + tupleDelimiter + "person" + tupleDelimiter + "the protagonist, revisiting his memories.\n" +
			"relation" + tupleDelimiter + "Ghost of Christmas Past" + tupleDelimiter + "Scrooge" + tupleDelimiter + "guidance,memory" + tupleDelimiter + "the ghost guides Scrooge through scenes of his past.\n" +
			completionDelimiter + "\n"

	case strings.Contains(chunkLower, "jacob marley") || strings.Contains(chunkLower, "spirits would visit"):
		if isContinue {
			return completionDelimiter + "\n"
		}
		return "" +
			"entity" + tupleDelimiter + "Jacob Marley" + tupleDelimiter + "person" + tupleDelimiter + "the ghost of Scrooge's late business partner.\n" +
			"entity" + tupleDelimiter + "Scrooge" + tupleDelimiter + "person" + tupleDelimiter + "the businessman warned by Marley's ghost.\n" +
			"relation" + tupleDelimiter + "Jacob Marley" + tupleDelimiter + "Scrooge" + tupleDelimiter + "warning,supernatural" + tupleDelimiter + "Marley's ghost warns Scrooge of three coming spirits.\n" +
			completionDelimiter + "\n"
	}

	// Generic fallback: synthesize one entity from the hash so unrecognized
	// chunks still round-trip through the parser.
	if isContinue {
		return completionDelimiter + "\n"
	}
	return "" +
		"entity" + tupleDelimiter + "Subject_" + hash[:8] + tupleDelimiter + "concept" + tupleDelimiter + "a synthesized entity for unrecognized chunk content.\n" +
		completionDelimiter + "\n"
}

// keyHash is a short hex digest used by mockCannedResponse for synthetic
// entity names — same crypto as cacheKey but truncated.
func keyHash(s string) string {
	h := sha256.Sum256([]byte(s))
	return hex.EncodeToString(h[:])
}

// --- compile-time interface contract assertion (also in tests) ---
var _ Provider = (*MockProvider)(nil)
var _ KVStore = (*MemoryKVStore)(nil)
