package main

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// extraction_test.go — exactly the 6 tests required by the s09 spec, plus
// a compile-time interface contract check.  Tests are numbered to match
// the spec for easy cross-reference.

// Compile-time assertion: *Extractor's exported method satisfies the
// internal contract.  Go has no formal "interface" for Extract but we use
// this anonymous interface as a sanity check against accidental signature
// drift.
type extractorContract interface {
	Extract(ctx context.Context, chunk Chunk) ([]Entity, []Relationship, error)
	Stats() Stats
}

var _ extractorContract = (*Extractor)(nil)

// (1) TestExtractionParsesDelimitedTuples — give parser a known delimited
//     string; assert correct entities + relations.
func TestExtractionParsesDelimitedTuples(t *testing.T) {
	in := "" +
		"entity" + tupleDelimiter + "Tokyo" + tupleDelimiter + "location" + tupleDelimiter + "the capital of Japan.\n" +
		"entity" + tupleDelimiter + "Mt. Fuji" + tupleDelimiter + "location" + tupleDelimiter + "the highest mountain in Japan.\n" +
		"relation" + tupleDelimiter + "Tokyo" + tupleDelimiter + "Mt. Fuji" + tupleDelimiter + "geography,landmark" + tupleDelimiter + "Mt. Fuji is visible from Tokyo on clear days.\n" +
		completionDelimiter + "\n"

	ents, rels, err := parseExtractionOutput(in)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(ents) != 2 {
		t.Fatalf("want 2 entities, got %d: %+v", len(ents), ents)
	}
	if ents[0].Name != "Tokyo" || ents[0].Type != "location" {
		t.Errorf("entity[0] wrong: %+v", ents[0])
	}
	if ents[1].Name != "Mt. Fuji" || ents[1].Type != "location" {
		t.Errorf("entity[1] wrong: %+v", ents[1])
	}
	if !strings.Contains(ents[0].Description, "capital of Japan") {
		t.Errorf("entity[0] description lost: %q", ents[0].Description)
	}

	if len(rels) != 1 {
		t.Fatalf("want 1 relation, got %d: %+v", len(rels), rels)
	}
	r := rels[0]
	if r.SrcID != "Tokyo" || r.TgtID != "Mt. Fuji" {
		t.Errorf("relation endpoints wrong: %+v", r)
	}
	if r.Keywords != "geography,landmark" {
		t.Errorf("relation keywords wrong: %q", r.Keywords)
	}
	if r.Weight != 1.0 {
		t.Errorf("relation weight should default to 1.0, got %v", r.Weight)
	}
}

// (2) TestExtractionGleaningAddsNewEntities — MockProvider returns 2 entities
//     on round 1, 1 NEW entity on round 2; assert gleaning loop accumulates
//     3 unique entities.
func TestExtractionGleaningAddsNewEntities(t *testing.T) {
	prov := NewMockProvider()
	// Use a chunk that triggers the Scrooge-Marley canned response on
	// round 1 (2 entities, 1 relation) and the Christmas-Eve-only response
	// on the gleaning continuation (1 NEW entity).
	chunk := Chunk{
		ContentDocID:    "doc",
		Content:         "Scrooge was a miserly businessman; Marley was his late partner.",
		ChunkOrderIndex: 0,
	}
	ext := NewExtractor(prov,
		WithGleaningRounds(1),
		WithCache(NewMemoryKVStore()),
	)

	ents, rels, err := ext.Extract(context.Background(), chunk)
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}
	if len(ents) != 3 {
		t.Errorf("want 3 unique entities (2 initial + 1 from gleaning), got %d: %+v", len(ents), ents)
	}
	// Assert the gleaning-only entity is present.
	hasChristmas := false
	for _, e := range ents {
		if e.Name == "Christmas Eve" {
			hasChristmas = true
			break
		}
	}
	if !hasChristmas {
		t.Errorf("gleaning round did not surface 'Christmas Eve' entity: %+v", ents)
	}
	if len(rels) < 1 {
		t.Errorf("want >= 1 relation, got %d", len(rels))
	}
	if prov.CallCount != 2 {
		t.Errorf("want 2 LLM calls (initial + 1 gleaning), got %d", prov.CallCount)
	}
}

// (3) TestExtractionCacheAvoidsDoubleCall — call Extract twice on same chunk;
//     assert second call hits cache (provider Call count is 1 if rounds=0).
func TestExtractionCacheAvoidsDoubleCall(t *testing.T) {
	prov := NewMockProvider()
	cache := NewMemoryKVStore()
	chunk := Chunk{
		ContentDocID:    "doc",
		Content:         "Scrooge and Marley were partners.",
		ChunkOrderIndex: 0,
	}
	ext := NewExtractor(prov, WithGleaningRounds(0), WithCache(cache))

	ctx := context.Background()
	if _, _, err := ext.Extract(ctx, chunk); err != nil {
		t.Fatalf("first Extract: %v", err)
	}
	firstCalls := prov.CallCount

	if _, _, err := ext.Extract(ctx, chunk); err != nil {
		t.Fatalf("second Extract: %v", err)
	}
	if prov.CallCount != firstCalls {
		t.Errorf("second Extract should not call LLM (cache hit); calls before=%d after=%d",
			firstCalls, prov.CallCount)
	}
	stats := ext.Stats()
	if stats.CacheHits != 1 {
		t.Errorf("want CacheHits=1, got %d", stats.CacheHits)
	}
	if stats.CacheMisses != 1 {
		t.Errorf("want CacheMisses=1, got %d", stats.CacheMisses)
	}
}

// (4) TestExtractionMalformedOutputFallsBack — provider returns malformed
//     delimited text (missing fields); parser returns empty without panic.
func TestExtractionMalformedOutputFallsBack(t *testing.T) {
	cases := map[string]string{
		"missing fields": "" +
			"entity" + tupleDelimiter + "JustAName\n" + // only 2 fields
			"relation" + tupleDelimiter + "A" + tupleDelimiter + "B\n" + // only 3 fields
			completionDelimiter,
		"wrong prefix": "" +
			"thingamajig" + tupleDelimiter + "X" + tupleDelimiter + "Y" + tupleDelimiter + "Z\n" +
			completionDelimiter,
		"empty input":   "",
		"whitespace":    "   \n\n  \t\n",
		"missing terminator": "" +
			"entity" + tupleDelimiter + "Tokyo" + tupleDelimiter + "location" + tupleDelimiter + "ok.\n",
		// "missing terminator" should still parse the one valid entity —
		// it's the legitimate-but-warning case.  We assert that below.
	}

	for name, in := range cases {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("parser panicked on %q: %v", name, r)
				}
			}()
			ents, rels, err := parseExtractionOutput(in)
			if err != nil {
				t.Fatalf("err on %q: %v", name, err)
			}
			switch name {
			case "missing terminator":
				if len(ents) != 1 || ents[0].Name != "Tokyo" {
					t.Errorf("missing-terminator should still parse the valid entity, got %+v", ents)
				}
			default:
				if len(ents) != 0 || len(rels) != 0 {
					t.Errorf("malformed input %q parsed something: ents=%+v rels=%+v",
						name, ents, rels)
				}
			}
		})
	}
}

// (5) TestExtractionMergesSourceIDsAcrossChunks — extract from chunk-1
//     (entity X with sources [chunk-1]) and chunk-2 (entity X with sources
//     [chunk-2]); assert dedup-by-name with merged SourceIDs=[chunk-1, chunk-2].
//
// The Extractor itself is per-chunk; cross-chunk merge is a CALLER
// responsibility (s10 owns it formally, but s09 demonstrates the
// invariant: extracting the same entity from two chunks yields two
// independent records each carrying ONE chunk ID, and a 5-line merge loop
// produces the desired union.
func TestExtractionMergesSourceIDsAcrossChunks(t *testing.T) {
	prov := NewMockProvider()
	// Override the response so both chunks extract the SAME entity name
	// "Shared" with different descriptions — that's what triggers the
	// merge.
	prov.ResponseOverride = map[string]string{
		"chunk-A-content": "" +
			"entity" + tupleDelimiter + "Shared" + tupleDelimiter + "concept" + tupleDelimiter + "version from A.\n" +
			completionDelimiter,
		"chunk-B-content": "" +
			"entity" + tupleDelimiter + "Shared" + tupleDelimiter + "concept" + tupleDelimiter + "version from B.\n" +
			completionDelimiter,
	}
	ext := NewExtractor(prov, WithGleaningRounds(0))
	ctx := context.Background()

	c1 := Chunk{ContentDocID: "doc", Content: "chunk-A-content", ChunkOrderIndex: 0}
	c2 := Chunk{ContentDocID: "doc", Content: "chunk-B-content", ChunkOrderIndex: 1}

	ents1, _, err := ext.Extract(ctx, c1)
	if err != nil {
		t.Fatalf("extract c1: %v", err)
	}
	ents2, _, err := ext.Extract(ctx, c2)
	if err != nil {
		t.Fatalf("extract c2: %v", err)
	}

	if len(ents1) != 1 || len(ents2) != 1 {
		t.Fatalf("each chunk should produce 1 entity, got %d / %d", len(ents1), len(ents2))
	}
	if got := ents1[0].SourceIDs; len(got) != 1 || got[0] != c1.ChunkID() {
		t.Errorf("c1 entity SourceIDs wrong: %+v (want [%s])", got, c1.ChunkID())
	}
	if got := ents2[0].SourceIDs; len(got) != 1 || got[0] != c2.ChunkID() {
		t.Errorf("c2 entity SourceIDs wrong: %+v (want [%s])", got, c2.ChunkID())
	}

	// Now do the cross-chunk merge that a downstream pipeline would do.
	merged := mergeEntitiesByName(append(ents1, ents2...))
	if len(merged) != 1 {
		t.Fatalf("after merge want 1 unique entity, got %d: %+v", len(merged), merged)
	}
	got := merged[0].SourceIDs
	if len(got) != 2 {
		t.Fatalf("merged SourceIDs should be 2, got %d: %+v", len(got), got)
	}
	want := map[string]bool{c1.ChunkID(): false, c2.ChunkID(): false}
	for _, id := range got {
		want[id] = true
	}
	for id, seen := range want {
		if !seen {
			t.Errorf("merged SourceIDs missing %s; got %+v", id, got)
		}
	}
}

// mergeEntitiesByName is the trivial caller-side merge demoed in test 5.
// In a real pipeline this lives in the orchestrator (s_full); s09 owns
// only the "extract from one chunk" half.
func mergeEntitiesByName(entities []Entity) []Entity {
	byName := make(map[string]*Entity)
	for i := range entities {
		e := entities[i]
		if existing, ok := byName[e.Name]; ok {
			existing.SourceIDs = append(existing.SourceIDs, e.SourceIDs...)
			// keep the LONGER description (matches upstream's heuristic)
			if len(e.Description) > len(existing.Description) {
				existing.Description = e.Description
			}
			continue
		}
		ec := e
		byName[ec.Name] = &ec
	}
	out := make([]Entity, 0, len(byName))
	for _, v := range byName {
		out = append(out, *v)
	}
	return out
}

// (6) TestExtractionRespectsContextCancel — provider sleeps; cancel ctx;
//     assert Extract returns context.Canceled wrapped.
func TestExtractionRespectsContextCancel(t *testing.T) {
	prov := NewMockProvider()
	prov.SleepMs = 200
	ext := NewExtractor(prov, WithGleaningRounds(0))
	ctx, cancel := context.WithCancel(context.Background())

	chunk := Chunk{
		ContentDocID:    "doc",
		Content:         "Anything triggers a sleep here.",
		ChunkOrderIndex: 0,
	}

	// Cancel after 30ms — provider sleep is 200ms so it must observe
	// ctx.Done before completing.
	time.AfterFunc(30*time.Millisecond, cancel)
	_, _, err := ext.Extract(ctx, chunk)
	if err == nil {
		t.Fatal("Extract should return error on canceled context")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err should wrap context.Canceled; got %v", err)
	}
}

// --- helper test: testdata/llm_responses files round-trip through parser ---
// (Not one of the 6 required tests; sanity-checks the testdata fixtures.)
func TestExtractionFixturesParse(t *testing.T) {
	dir := filepath.Join("testdata", "llm_responses")
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Skipf("no testdata dir: %v", err)
	}
	for _, ent := range entries {
		if ent.IsDir() || !strings.HasSuffix(ent.Name(), ".txt") {
			continue
		}
		t.Run(ent.Name(), func(t *testing.T) {
			b, err := os.ReadFile(filepath.Join(dir, ent.Name()))
			if err != nil {
				t.Fatalf("read: %v", err)
			}
			ents, rels, err := parseExtractionOutput(string(b))
			if err != nil {
				t.Fatalf("parse: %v", err)
			}
			if len(ents) == 0 && len(rels) == 0 {
				t.Errorf("fixture %s parsed to nothing", ent.Name())
			}
		})
	}
}
