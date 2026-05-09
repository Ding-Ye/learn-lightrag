package main

import (
	"context"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
)

// main is the s07 CLI demo: spin up THREE separate CosineIndex instances —
// one each for chunks, entities, relations — seed them with deterministic
// random vectors, run a sample query against each, and print the top-3.
//
// The point is to make visible the upstream pattern of "three vdbs, one
// store impl" without baking any pipeline coupling into the s07 module.
func main() {
	dir := flag.String("dir", "./lightrag-data", "where to put the vdb_*.json files")
	flag.Parse()

	ctx := context.Background()
	if err := run(ctx, *dir); err != nil {
		fmt.Fprintf(os.Stderr, "s07 demo failed: %v\n", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, dir string) error {
	const dim = 8
	// Each demo namespace gets its own deterministic RNG seed so re-runs
	// emit byte-identical output.  Seeds chosen arbitrarily; only the
	// determinism matters.
	demos := []struct {
		ns        Namespace
		idPrefix  string // singular form, used to label records
		seed      int64
		count     int
		queryText string // ID of the record whose vector becomes our query
	}{
		{NamespaceChunks, "chunk", 42, 6, "chunk-2"},
		{NamespaceEntities, "entity", 137, 5, "entity-1"},
		{NamespaceRelations, "relation", 271, 4, "relation-3"},
	}
	for _, d := range demos {
		fmt.Printf("=== namespace: %s ===\n", d.ns)
		idx, err := NewCosineIndex(dir, d.ns)
		if err != nil {
			return fmt.Errorf("ns=%s: %w", d.ns, err)
		}

		// Build deterministic records from a fixed-seed RNG.
		rng := rand.New(rand.NewSource(d.seed))
		records := make([]VectorRecord, 0, d.count)
		var queryVec []float32
		for i := 0; i < d.count; i++ {
			id := fmt.Sprintf("%s-%d", d.idPrefix, i) // chunk-0, entity-0, relation-0...
			vec := randUnitVector(rng, dim)
			records = append(records, VectorRecord{
				ID:       id,
				Vector:   vec,
				Metadata: map[string]any{"label": id, "ord": i},
			})
			if id == d.queryText {
				queryVec = vec
			}
		}
		if err := idx.Upsert(ctx, records); err != nil {
			return fmt.Errorf("ns=%s upsert: %w", d.ns, err)
		}

		// If the requested query label wasn't found, fall back to the first
		// seed vector so the demo always has SOMETHING to query against.
		if queryVec == nil {
			queryVec = records[0].Vector
		}

		hits, err := idx.Query(ctx, queryVec, 3, 0.0)
		if err != nil {
			return fmt.Errorf("ns=%s query: %w", d.ns, err)
		}
		fmt.Printf("count=%d  query=%s  top-3:\n", idx.Len(), d.queryText)
		for rank, h := range hits {
			fmt.Printf("  %d. id=%-14s score=%.4f\n", rank+1, h.ID, h.Score)
		}

		if err := idx.Persist(ctx); err != nil {
			return fmt.Errorf("ns=%s persist: %w", d.ns, err)
		}
		fmt.Printf("persisted -> %s/vdb_%s.json\n\n", dir, d.ns)
	}
	fmt.Println("All three indices built and persisted.")
	fmt.Println("Re-run with the same -dir to confirm Load round-trips.")
	return nil
}

// randUnitVector emits a deterministic unit-norm vector using the supplied
// RNG.  We intentionally L2-normalize so downstream cosine scores live in a
// meaningful range — same trick as s06's MockEmbedder.
func randUnitVector(rng *rand.Rand, dim int) []float32 {
	v := make([]float32, dim)
	var sumSq float64
	for i := range v {
		// rng.NormFloat64 produces standard normals; centered around 0 so
		// the resulting vector points in a random direction on S^(dim-1)
		// after normalization.
		x := rng.NormFloat64()
		v[i] = float32(x)
		sumSq += x * x
	}
	if sumSq == 0 {
		return v
	}
	norm := float32(math.Sqrt(sumSq))
	for i := range v {
		v[i] /= norm
	}
	return v
}
