package main

import (
	"context"
	"strings"
	"testing"
)

// query_test.go covers the seven required tests for s11.  Every test
// constructs a fresh seeded pipeline so behaviour is deterministic.  All
// tests stay offline — MockProvider + MockEmbedder, no network.

// =============================================================================
// 1. Naive mode returns top chunks for a known query.
// =============================================================================

func TestQueryNaiveReturnsTopChunks(t *testing.T) {
	ctx := context.Background()
	pipe, err := buildSeededPipeline(ctx)
	if err != nil {
		t.Fatalf("seed: %v", err)
	}
	// Use a query that names the codex directly so chunk-2 / chunk-3 (which
	// also mention "Hartwell Codex") rise to the top of cosine similarity.
	res, err := pipe.Query(ctx, "Who discovered the Hartwell Codex manuscript?", QueryParam{
		Mode: ModeNaive, ChunkTopK: 3, MaxTotalTokens: 1000,
	})
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if res.Mode != ModeNaive {
		t.Errorf("Mode = %v, want %v", res.Mode, ModeNaive)
	}
	if len(res.References) == 0 {
		t.Fatalf("References empty")
	}
	// chunk-2 mentions "Hartwell Codex" and "discovered" — the strongest
	// semantic overlap with the query under MockEmbedder's per-word hash.
	found := false
	for _, id := range res.References {
		if id == "chunk-2" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("chunk-2 not in References = %v", res.References)
	}
}

// =============================================================================
// 2. Local mode passes LOW-level keywords (entities) to VDBEntities.
// =============================================================================

// recordingVectorStore wraps a memVectorStore and records the queries it
// sees so a test can assert which embeddings reached which index.
type recordingVectorStore struct {
	inner   *memVectorStore
	queries [][]float32
	calls   int
}

func newRecordingVec() *recordingVectorStore {
	return &recordingVectorStore{inner: newMemVec()}
}

func (r *recordingVectorStore) Upsert(ctx context.Context, records []VectorRecord) error {
	return r.inner.Upsert(ctx, records)
}
func (r *recordingVectorStore) Query(ctx context.Context, query []float32, topK int, threshold float32) ([]VectorHit, error) {
	r.calls++
	r.queries = append(r.queries, query)
	return r.inner.Query(ctx, query, topK, threshold)
}
func (r *recordingVectorStore) Delete(ctx context.Context, ids []string) error {
	return r.inner.Delete(ctx, ids)
}
func (r *recordingVectorStore) Persist(ctx context.Context) error { return r.inner.Persist(ctx) }

// vectorMatches returns true if any vector in haystack equals (within float
// epsilon) needle.  Used to confirm a specific embedding (e.g. of a low-
// level keyword) was sent to a specific store.
func vectorMatches(haystack [][]float32, needle []float32) bool {
	const eps = 1e-6
	for _, v := range haystack {
		if len(v) != len(needle) {
			continue
		}
		ok := true
		for i := range v {
			d := float64(v[i] - needle[i])
			if d < -eps || d > eps {
				ok = false
				break
			}
		}
		if ok {
			return true
		}
	}
	return false
}

func TestQueryLocalUsesLowLevelKeywords(t *testing.T) {
	ctx := context.Background()
	pipe, err := buildSeededPipeline(ctx)
	if err != nil {
		t.Fatalf("seed: %v", err)
	}
	// Wrap the entity & relation stores in recorders so we see what each got.
	entRec := newRecordingVec()
	relRec := newRecordingVec()
	// Re-upsert: copy every entity/relation record from the seeded inner
	// stores into the recorders so the test isn't testing an empty index.
	if err := copyVecRecords(ctx, pipe.VDBEnts, entRec); err != nil {
		t.Fatalf("copy ents: %v", err)
	}
	if err := copyVecRecords(ctx, pipe.VDBRels, relRec); err != nil {
		t.Fatalf("copy rels: %v", err)
	}
	pipe.VDBEnts = entRec
	pipe.VDBRels = relRec

	// MockProvider's canonicalKeywords picks up "hartwell" → low-level
	// includes "Eleanor Hartwell".  Embed that keyword and verify VDBEnts
	// (NOT VDBRels) saw the embedding.
	emb := pipe.Embedder.(*MockEmbedder)
	keywordVec, _ := emb.Embed(ctx, []string{"Eleanor Hartwell"})
	if _, err := pipe.Query(ctx, "Tell me about Hartwell.", QueryParam{
		Mode: ModeLocal, TopK: 5, ChunkTopK: 5, MaxTotalTokens: 1000,
	}); err != nil {
		t.Fatalf("Query: %v", err)
	}
	if entRec.calls == 0 {
		t.Fatalf("VDBEntities was not queried")
	}
	if !vectorMatches(entRec.queries, keywordVec[0]) {
		t.Errorf("VDBEntities did not see the 'Eleanor Hartwell' keyword embedding (got %d queries)", entRec.calls)
	}
	if relRec.calls != 0 {
		t.Errorf("VDBRelations was queried %d times in local mode (want 0)", relRec.calls)
	}
}

// copyVecRecords copies every record from src to dst.  Implementation peeks
// at the *memVectorStore underneath via a type assertion; that's fine here
// because we control both sides.
func copyVecRecords(ctx context.Context, src VectorStore, dst VectorStore) error {
	mem, ok := src.(*memVectorStore)
	if !ok {
		return nil
	}
	mem.mu.RLock()
	defer mem.mu.RUnlock()
	cp := make([]VectorRecord, len(mem.records))
	copy(cp, mem.records)
	return dst.Upsert(ctx, cp)
}

// =============================================================================
// 3. Global mode passes HIGH-level keywords (themes) to VDBRelations.
// =============================================================================

func TestQueryGlobalUsesHighLevelKeywords(t *testing.T) {
	ctx := context.Background()
	pipe, err := buildSeededPipeline(ctx)
	if err != nil {
		t.Fatalf("seed: %v", err)
	}
	entRec := newRecordingVec()
	relRec := newRecordingVec()
	if err := copyVecRecords(ctx, pipe.VDBEnts, entRec); err != nil {
		t.Fatalf("copy ents: %v", err)
	}
	if err := copyVecRecords(ctx, pipe.VDBRels, relRec); err != nil {
		t.Fatalf("copy rels: %v", err)
	}
	pipe.VDBEnts = entRec
	pipe.VDBRels = relRec

	// "What discoveries arose around the codex?" → MockProvider extracts
	// "discovery" as a high-level keyword.
	emb := pipe.Embedder.(*MockEmbedder)
	hlVec, _ := emb.Embed(ctx, []string{"discovery"})
	if _, err := pipe.Query(ctx, "What discoveries arose around the codex?", QueryParam{
		Mode: ModeGlobal, TopK: 5, ChunkTopK: 5, MaxTotalTokens: 1000,
	}); err != nil {
		t.Fatalf("Query: %v", err)
	}
	if relRec.calls == 0 {
		t.Fatalf("VDBRelations was not queried in global mode")
	}
	if !vectorMatches(relRec.queries, hlVec[0]) {
		t.Errorf("VDBRelations did not see the 'discovery' high-level keyword embedding")
	}
	if entRec.calls != 0 {
		t.Errorf("VDBEntities was queried %d times in global mode (want 0)", entRec.calls)
	}
}

// =============================================================================
// 4. Hybrid dedupes chunks across local + global retrievals.
// =============================================================================

func TestQueryHybridDedupesChunks(t *testing.T) {
	ctx := context.Background()
	pipe, err := buildSeededPipeline(ctx)
	if err != nil {
		t.Fatalf("seed: %v", err)
	}
	res, err := pipe.Query(ctx, "Who discovered the Hartwell Codex and verified it?", QueryParam{
		Mode: ModeHybrid, TopK: 5, ChunkTopK: 6, MaxTotalTokens: 1500,
	})
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	seen := map[string]int{}
	for _, id := range res.References {
		seen[id]++
	}
	for id, n := range seen {
		if n > 1 {
			t.Errorf("chunk %q appears %d times in References (want 1)", id, n)
		}
	}
}

// =============================================================================
// 5. MaxTotalTokens caps the assembled prompt size.
// =============================================================================

func TestQueryRespectsMaxTotalTokens(t *testing.T) {
	ctx := context.Background()
	pipe, err := buildSeededPipeline(ctx)
	if err != nil {
		t.Fatalf("seed: %v", err)
	}
	prov := pipe.Provider.(*MockProvider)
	const budget = 50
	if _, err := pipe.Query(ctx, "Who is Eleanor Hartwell and what work is she known for?", QueryParam{
		Mode: ModeHybrid, TopK: 5, ChunkTopK: 6,
		MaxEntityTokens: 20, MaxRelationTokens: 20, MaxTotalTokens: budget,
	}); err != nil {
		t.Fatalf("Query: %v", err)
	}
	wc := tokenCount(prov.LastSystem)
	if wc > budget+5 { // +5 envelope for header words; conservative
		t.Errorf("system prompt = %d words, want <= ~%d (budget %d)", wc, budget+5, budget)
	}
}

// =============================================================================
// 6. Every successful query returns a non-empty References list.
// =============================================================================

func TestQueryReturnsCitedChunkIDs(t *testing.T) {
	ctx := context.Background()
	pipe, err := buildSeededPipeline(ctx)
	if err != nil {
		t.Fatalf("seed: %v", err)
	}
	for _, m := range []QueryMode{ModeNaive, ModeLocal, ModeGlobal, ModeHybrid} {
		res, err := pipe.Query(ctx, "Who is Eleanor Hartwell and what is the codex?", QueryParam{
			Mode: m, TopK: 5, ChunkTopK: 4, MaxTotalTokens: 1000,
		})
		if err != nil {
			t.Errorf("[%s] Query: %v", m, err)
			continue
		}
		if len(res.References) == 0 {
			t.Errorf("[%s] References is empty", m)
		}
	}
}

// =============================================================================
// 7. Keyword extraction parses both arrays from JSON.
// =============================================================================

func TestKeywordExtractionParsesBothLevels(t *testing.T) {
	ctx := context.Background()
	prov := NewMockProvider()
	prov.FixedKeywords = `{"high_level_keywords": ["medicine", "policy"], "low_level_keywords": ["Eleanor Hartwell", "Hartwell Codex"]}`
	hl, ll, err := extractKeywords(ctx, prov, "anything")
	if err != nil {
		t.Fatalf("extractKeywords: %v", err)
	}
	if len(hl) != 2 || hl[0] != "medicine" || hl[1] != "policy" {
		t.Errorf("hl = %v, want [medicine policy]", hl)
	}
	if len(ll) != 2 || ll[0] != "Eleanor Hartwell" || ll[1] != "Hartwell Codex" {
		t.Errorf("ll = %v, want [Eleanor Hartwell Hartwell Codex]", ll)
	}
	// Bonus: markdown-fence drift should still parse.
	prov2 := NewMockProvider()
	prov2.FixedKeywords = "```json\n{\"high_level_keywords\": [\"a\"], \"low_level_keywords\": [\"b\"]}\n```"
	hl2, ll2, err := extractKeywords(ctx, prov2, "x")
	if err != nil {
		t.Fatalf("fenced extractKeywords: %v", err)
	}
	if len(hl2) != 1 || hl2[0] != "a" {
		t.Errorf("fenced hl = %v, want [a]", hl2)
	}
	if len(ll2) != 1 || ll2[0] != "b" {
		t.Errorf("fenced ll = %v, want [b]", ll2)
	}
	// Bonus 2: surrounding prose.
	prov3 := NewMockProvider()
	prov3.FixedKeywords = "Sure! Here is the JSON:\n{\"high_level_keywords\": [\"theme\"], \"low_level_keywords\": [\"thing\"]}\nThanks."
	hl3, ll3, err := extractKeywords(ctx, prov3, "x")
	if err != nil {
		t.Fatalf("prose extractKeywords: %v", err)
	}
	if len(hl3) != 1 || hl3[0] != "theme" {
		t.Errorf("prose hl = %v", hl3)
	}
	if len(ll3) != 1 || ll3[0] != "thing" {
		t.Errorf("prose ll = %v", ll3)
	}
}

// =============================================================================
// 8 (compile-time): Pipeline interface contract.
// =============================================================================
//
// Asserts that every store the Pipeline holds satisfies its interface.  If
// any of the stub types diverge from the catalog, this fails to compile.

func TestPipelineInterfaceContract(t *testing.T) {
	var _ KVStore = (*memKVStore)(nil)
	var _ VectorStore = (*memVectorStore)(nil)
	var _ GraphStore = (*memGraphStore)(nil)
	var _ Provider = (*MockProvider)(nil)
	var _ EmbeddingProvider = (*MockEmbedder)(nil)
	var _ VectorStore = (*recordingVectorStore)(nil)
	// touch one method per interface so the test isn't a no-op at runtime.
	if !strings.Contains(string(ModeHybrid), "hybrid") {
		t.Errorf("ModeHybrid missing 'hybrid' substring")
	}
}
