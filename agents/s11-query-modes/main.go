package main

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"os"
	"strings"
)

// main.go is the s11 CLI demo.  Seeds the in-memory stores with a tiny
// fictional corpus (the recurring "Eleanor Hartwell / Whitehall Museum"
// world from earlier sessions), then runs one — or all four — query modes
// and prints retrievals + final answer side by side.
//
// Usage:
//   go run . -mode hybrid -q "Who restored the Hartwell Codex?"
//   go run . -mode all -q "What conflicts arose around the codex?"

func main() {
	mode := flag.String("mode", "hybrid", "query mode: naive | local | global | hybrid | all")
	q := flag.String("q", "Who is Eleanor Hartwell and what work is she known for?", "user query")
	flag.Parse()
	if err := runDemo(context.Background(), *mode, *q); err != nil {
		fmt.Fprintf(os.Stderr, "s11 demo failed: %v\n", err)
		os.Exit(1)
	}
}

func runDemo(ctx context.Context, mode, q string) error {
	pipe, err := buildSeededPipeline(ctx)
	if err != nil {
		return err
	}

	param := QueryParam{
		TopK:              5,
		ChunkTopK:         5,
		MaxEntityTokens:   200,
		MaxRelationTokens: 200,
		MaxTotalTokens:    1200,
	}

	modes := []QueryMode{}
	switch mode {
	case "naive":
		modes = []QueryMode{ModeNaive}
	case "local":
		modes = []QueryMode{ModeLocal}
	case "global":
		modes = []QueryMode{ModeGlobal}
	case "hybrid":
		modes = []QueryMode{ModeHybrid}
	case "all":
		modes = []QueryMode{ModeNaive, ModeLocal, ModeGlobal, ModeHybrid}
	default:
		return fmt.Errorf("unknown -mode: %s (want naive|local|global|hybrid|all)", mode)
	}

	fmt.Printf("=== s11: query=%q ===\n\n", q)
	for _, m := range modes {
		param.Mode = m
		res, err := pipe.Query(ctx, q, param)
		if err != nil {
			fmt.Printf("[%s] FAILED: %v\n\n", m, err)
			continue
		}
		fmt.Printf("[%s]\n", m)
		fmt.Printf("  references: %v\n", res.References)
		fmt.Printf("  answer    : %s\n", oneLine(res.Content))
		fmt.Println(strings.Repeat("-", 60))
	}
	return nil
}

func oneLine(s string) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) > 220 {
		s = s[:217] + "..."
	}
	return s
}

// buildSeededPipeline constructs the in-memory pipeline + seeds 6 chunks,
// 5 entities, and 4 relations into the stores.  All embeddings use the
// MockEmbedder so behaviour is deterministic.
func buildSeededPipeline(ctx context.Context) (*Pipeline, error) {
	emb := NewMockEmbedder(8)
	prov := NewMockProvider()

	pipe := &Pipeline{
		Provider:  prov,
		Embedder:  emb,
		VDBChunks: newMemVec(),
		VDBEnts:   newMemVec(),
		VDBRels:   newMemVec(),
		KV:        newMemKV(),
		Graph:     newMemGraph(),
	}

	// Chunks (id → content).  References mention different facets of the
	// world so naive vector search finds different ones depending on query.
	chunks := []chunkRow{
		{"chunk-1", "Eleanor Hartwell is a senior archivist at the Whitehall Museum, restoring medieval manuscripts."},
		{"chunk-2", "Eleanor Hartwell discovered the Hartwell Codex during a routine cataloguing project."},
		{"chunk-3", "The Hartwell Codex predates the Magna Carta according to a paper by Eleanor Hartwell in the Journal of Medieval Studies."},
		{"chunk-4", "Cambridge historians collaborated with Hartwell to verify the codex's provenance."},
		{"chunk-5", "A parliamentary inquiry into cultural-property repatriation called Hartwell as a key witness."},
		{"chunk-6", "Eleanor Hartwell mentored younger archivists who now lead European institutions."},
	}
	kvItems := map[string]map[string]any{}
	chunkRecs := make([]VectorRecord, 0, len(chunks))
	for _, c := range chunks {
		kvItems[c.ID] = map[string]any{"content": c.Content}
		v, err := emb.Embed(ctx, []string{c.Content})
		if err != nil {
			return nil, err
		}
		chunkRecs = append(chunkRecs, VectorRecord{ID: c.ID, Vector: v[0], Metadata: map[string]any{"chunk_id": c.ID}})
	}
	if err := pipe.KV.Upsert(ctx, kvItems); err != nil {
		return nil, err
	}
	if err := pipe.VDBChunks.Upsert(ctx, chunkRecs); err != nil {
		return nil, err
	}

	// Entities — name, type, description, source chunks.
	entities := []Entity{
		{Name: "Eleanor Hartwell", Type: "person", Description: "Senior archivist at the Whitehall Museum.", SourceIDs: []string{"chunk-1", "chunk-2", "chunk-6"}},
		{Name: "Whitehall Museum", Type: "organization", Description: "Museum employing Eleanor Hartwell.", SourceIDs: []string{"chunk-1"}},
		{Name: "Hartwell Codex", Type: "artifact", Description: "Medieval manuscript discovered by Hartwell.", SourceIDs: []string{"chunk-2", "chunk-3", "chunk-4"}},
		{Name: "Cambridge", Type: "organization", Description: "University whose historians verified the codex.", SourceIDs: []string{"chunk-4"}},
		{Name: "Parliamentary Inquiry", Type: "event", Description: "Cultural-property repatriation inquiry where Hartwell testified.", SourceIDs: []string{"chunk-5"}},
	}
	entRecs := make([]VectorRecord, 0, len(entities))
	for _, e := range entities {
		if err := pipe.Graph.UpsertNode(ctx, e); err != nil {
			return nil, err
		}
		v, err := emb.Embed(ctx, []string{e.Name + " " + e.Description})
		if err != nil {
			return nil, err
		}
		entRecs = append(entRecs, VectorRecord{
			ID: e.Name, Vector: v[0],
			Metadata: map[string]any{"entity_id": e.Name, "type": e.Type},
		})
	}
	if err := pipe.VDBEnts.Upsert(ctx, entRecs); err != nil {
		return nil, err
	}

	// Relations — directional pairs with shared chunk provenance.
	rels := []Relationship{
		{SrcID: "Eleanor Hartwell", TgtID: "Whitehall Museum", Keywords: "employment", Description: "Hartwell works at the Whitehall Museum.", Weight: 1.0, SourceIDs: []string{"chunk-1"}},
		{SrcID: "Eleanor Hartwell", TgtID: "Hartwell Codex", Keywords: "discovery", Description: "Hartwell discovered the codex.", Weight: 1.0, SourceIDs: []string{"chunk-2", "chunk-3"}},
		{SrcID: "Cambridge", TgtID: "Hartwell Codex", Keywords: "verification", Description: "Cambridge historians verified the codex.", Weight: 0.9, SourceIDs: []string{"chunk-4"}},
		{SrcID: "Eleanor Hartwell", TgtID: "Parliamentary Inquiry", Keywords: "testimony", Description: "Hartwell testified in the inquiry.", Weight: 0.8, SourceIDs: []string{"chunk-5"}},
	}
	relRecs := make([]VectorRecord, 0, len(rels))
	for _, r := range rels {
		if err := pipe.Graph.UpsertEdge(ctx, r); err != nil {
			return nil, err
		}
		text := r.Keywords + " " + r.Description
		v, err := emb.Embed(ctx, []string{text})
		if err != nil {
			return nil, err
		}
		relRecs = append(relRecs, VectorRecord{
			ID:     r.SrcID + "->" + r.TgtID,
			Vector: v[0],
			Metadata: map[string]any{
				"src_id": r.SrcID, "tgt_id": r.TgtID,
				"keywords": r.Keywords, "description": r.Description,
				"source_ids": r.SourceIDs, "weight": r.Weight,
			},
		})
	}
	if err := pipe.VDBRels.Upsert(ctx, relRecs); err != nil {
		return nil, err
	}

	return pipe, nil
}

// =============================================================================
// MockProvider — deterministic LLM stand-in
// =============================================================================
//
// Emits a JSON keyword payload when it sees the keyword-extraction system
// prompt; emits a canned synthesis answer otherwise.  Tracks call count so
// tests can assert.

type MockProvider struct {
	Calls int
	// LastSystem is the system prompt of the last Complete call.  Tests use
	// this for assertions about prompt assembly.
	LastSystem string
	// FixedResponse, if non-empty, overrides the default canned answer for
	// non-keyword calls.  Used by TestKeywordExtractionParsesBothLevels.
	FixedResponse string
	// FixedKeywords, if non-empty, overrides the default keyword JSON.
	FixedKeywords string
}

// NewMockProvider returns a fresh MockProvider.
func NewMockProvider() *MockProvider { return &MockProvider{} }

// Complete implements Provider.  Branch on the system prompt: if it looks
// like the keyword-extraction system prompt, return a JSON object; else,
// return the canned synthesis answer.
func (m *MockProvider) Complete(ctx context.Context, req CompleteRequest) (CompleteResponse, error) {
	m.Calls++
	m.LastSystem = req.System
	if err := ctx.Err(); err != nil {
		return CompleteResponse{}, err
	}
	if strings.Contains(req.System, "keyword extraction agent") {
		if m.FixedKeywords != "" {
			return CompleteResponse{Text: m.FixedKeywords}, nil
		}
		// Pull a few content words from the user message.
		userText := ""
		for _, msg := range req.Messages {
			if msg.Role == "user" {
				userText = msg.Content
			}
		}
		hl, ll := canonicalKeywords(userText)
		jsonOut := fmt.Sprintf(`{"high_level_keywords": [%s], "low_level_keywords": [%s]}`,
			quoteList(hl), quoteList(ll))
		return CompleteResponse{Text: jsonOut}, nil
	}
	if m.FixedResponse != "" {
		return CompleteResponse{Text: m.FixedResponse}, nil
	}
	return CompleteResponse{Text: "Eleanor Hartwell is a senior archivist at the Whitehall Museum who discovered the Hartwell Codex."}, nil
}

func canonicalKeywords(userText string) (hl, ll []string) {
	low := strings.ToLower(userText)
	if strings.Contains(low, "eleanor") || strings.Contains(low, "hartwell") {
		ll = append(ll, "Eleanor Hartwell")
	}
	if strings.Contains(low, "codex") {
		ll = append(ll, "Hartwell Codex")
	}
	if strings.Contains(low, "museum") || strings.Contains(low, "whitehall") {
		ll = append(ll, "Whitehall Museum")
	}
	if strings.Contains(low, "discover") || strings.Contains(low, "discovery") {
		hl = append(hl, "discovery")
	}
	if strings.Contains(low, "conflict") || strings.Contains(low, "inquiry") || strings.Contains(low, "dispute") {
		hl = append(hl, "conflict")
	}
	if strings.Contains(low, "verif") || strings.Contains(low, "authentic") {
		hl = append(hl, "verification")
	}
	if len(ll) == 0 {
		ll = append(ll, "Eleanor Hartwell")
	}
	if len(hl) == 0 {
		hl = append(hl, "discovery")
	}
	return hl, ll
}

func quoteList(xs []string) string {
	parts := make([]string, len(xs))
	for i, x := range xs {
		parts[i] = `"` + x + `"`
	}
	return strings.Join(parts, ", ")
}

// =============================================================================
// MockEmbedder — deterministic hash-based embedder
// =============================================================================
//
// Embed returns a vector whose components are derived from a SHA-256 of the
// input plus a per-dimension bias.  Same input → same vector.  Different
// inputs that share lots of words have correlated vectors so VectorStore
// cosine similarity behaves usefully on the seeded corpus.

type MockEmbedder struct{ dim int }

// NewMockEmbedder returns an embedder with the requested dim.
func NewMockEmbedder(dim int) *MockEmbedder {
	if dim <= 0 {
		dim = 8
	}
	return &MockEmbedder{dim: dim}
}

func (e *MockEmbedder) Dim() int { return e.dim }

func (e *MockEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		out[i] = embedOne(t, e.dim)
	}
	return out, nil
}

// embedOne hashes each whitespace-word of t into a per-dim slot.  The slot
// receives accumulated weight from the word's hash and a small bias.  This
// produces correlated vectors for texts that share words and orthogonal-ish
// vectors for texts that don't.
func embedOne(t string, dim int) []float32 {
	v := make([]float32, dim)
	if dim == 0 {
		return v
	}
	t = strings.ToLower(t)
	words := strings.Fields(t)
	if len(words) == 0 {
		return v
	}
	for _, w := range words {
		h := sha256.Sum256([]byte(w))
		for i := 0; i < dim; i++ {
			// 4 bytes of the hash → uint32 → float in [-1, 1).
			off := (i * 4) % (len(h) - 4)
			x := binary.BigEndian.Uint32(h[off : off+4])
			v[i] += float32(x)/float32(math.MaxUint32)*2 - 1
		}
	}
	// L2-normalize so cosine similarity behaves well.
	var norm float64
	for _, x := range v {
		norm += float64(x) * float64(x)
	}
	if norm == 0 {
		return v
	}
	inv := float32(1.0 / math.Sqrt(norm))
	for i := range v {
		v[i] *= inv
	}
	return v
}

// --- compile-time interface assertions ---------------------------------------

var (
	_ Provider          = (*MockProvider)(nil)
	_ EmbeddingProvider = (*MockEmbedder)(nil)
)
