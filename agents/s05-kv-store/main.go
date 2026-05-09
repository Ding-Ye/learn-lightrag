package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"sort"
)

// CLI demo for s05.  Walks through Upsert -> Get -> Delete -> FilterMissing
// -> Persist on a single namespace, then prints the on-disk file path so the
// learner can inspect the JSON output.  Re-running the command on the same
// (dir, namespace) pair re-loads the previously persisted records.
//
//   go run . [-dir ./lightrag-data] [-namespace demo]
func main() {
	var dir, ns string
	flag.StringVar(&dir, "dir", "./lightrag-data", "directory holding kv_<namespace>.json")
	flag.StringVar(&ns, "namespace", "demo", "kv namespace (filename prefix)")
	flag.Parse()

	if err := run(dir, ns); err != nil {
		log.Fatalf("s05 demo failed: %v", err)
	}
}

func run(dir, ns string) error {
	ctx := context.Background()

	store, err := NewJSONKVStore(dir, ns)
	if err != nil {
		return fmt.Errorf("new store: %w", err)
	}
	fmt.Printf("== s05 kv-store demo ==\n  dir       = %s\n  namespace = %s\n  on-disk   = %s\n  loaded    = %d record(s)\n\n",
		dir, ns, store.filePath(), store.Len())

	// --- Upsert 5 demo records ---
	items := map[string]map[string]any{
		"chunk-0": {"content": "scrooge was a tight-fisted hand at the grindstone", "tokens": 9},
		"chunk-1": {"content": "marley was dead, to begin with", "tokens": 7},
		"chunk-2": {"content": "the ghost of christmas past", "tokens": 6},
		"chunk-3": {"content": "tiny tim observed the day", "tokens": 5},
		"chunk-4": {"content": "god bless us, every one", "tokens": 5},
	}
	if err := store.Upsert(ctx, items); err != nil {
		return err
	}
	fmt.Printf("  Upsert  -> %d records staged in memory\n", len(items))

	// --- Get them back ---
	for _, id := range sortedKeys(items) {
		rec, ok, err := store.Get(ctx, id)
		if err != nil {
			return err
		}
		if !ok {
			return fmt.Errorf("missing id %q after Upsert", id)
		}
		fmt.Printf("    %-9s tokens=%v  content=%q\n", id, rec["tokens"], rec["content"])
	}

	// --- Delete two ---
	toDel := []string{"chunk-1", "chunk-3"}
	if err := store.Delete(ctx, toDel); err != nil {
		return err
	}
	fmt.Printf("\n  Delete  -> removed %v\n  remaining = %d\n", toDel, store.Len())

	// --- FilterMissing demo: which IDs are NOT in the store? ---
	candidate := []string{"chunk-0", "chunk-1", "chunk-3", "chunk-9", "chunk-42"}
	missing, err := store.FilterMissing(ctx, candidate)
	if err != nil {
		return err
	}
	fmt.Printf("\n  FilterMissing(%v)\n     -> %v   (the IDs s09 would still need to extract)\n", candidate, missing)

	// --- Persist & confirm the on-disk artifact ---
	if err := store.Persist(ctx); err != nil {
		return err
	}
	fmt.Printf("\n  Persist -> wrote %s\n  (rerun this command and watch `loaded = %d` reflect the persisted records)\n",
		store.filePath(), store.Len())
	return nil
}

func sortedKeys(m map[string]map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
