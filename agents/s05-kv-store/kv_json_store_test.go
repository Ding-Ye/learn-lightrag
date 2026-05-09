package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"testing"
)

// Compile-time interface contract — if JSONKVStore drifts away from KVStore
// the test suite refuses to compile.  No runtime cost.
func TestKVStoreInterfaceContract(t *testing.T) {
	var _ KVStore = (*JSONKVStore)(nil)
}

// Round-trip the basic Upsert -> Get path.  Asserts the returned map equals
// what we put in (a defensive copy is fine; values must match).
func TestKVUpsertGetRoundTrip(t *testing.T) {
	ctx := context.Background()
	store, err := NewJSONKVStore(t.TempDir(), "round-trip")
	if err != nil {
		t.Fatalf("new store: %v", err)
	}

	in := map[string]any{"content": "hello", "tokens": 1}
	if err := store.Upsert(ctx, map[string]map[string]any{"k1": in}); err != nil {
		t.Fatalf("upsert: %v", err)
	}

	out, ok, err := store.Get(ctx, "k1")
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	if !ok {
		t.Fatal("expected hit, got miss")
	}
	if got, want := fmt.Sprint(out["content"]), "hello"; got != want {
		t.Errorf("content = %q, want %q", got, want)
	}
	if got, want := fmt.Sprint(out["tokens"]), "1"; got != want {
		t.Errorf("tokens = %q, want %q", got, want)
	}

	// Defensive-copy check: mutating the returned map must not affect the
	// store's internal state.
	out["content"] = "tampered"
	again, _, _ := store.Get(ctx, "k1")
	if again["content"] == "tampered" {
		t.Error("Get returned a live reference; expected a defensive copy")
	}
}

// FilterMissing must return exactly the subset of input ids NOT in the
// store, regardless of order.
func TestKVFilterMissingPartial(t *testing.T) {
	ctx := context.Background()
	store, err := NewJSONKVStore(t.TempDir(), "filter")
	if err != nil {
		t.Fatalf("new store: %v", err)
	}

	if err := store.Upsert(ctx, map[string]map[string]any{
		"a": {"v": 1},
		"b": {"v": 2},
	}); err != nil {
		t.Fatalf("upsert: %v", err)
	}

	missing, err := store.FilterMissing(ctx, []string{"a", "b", "c", "d"})
	if err != nil {
		t.Fatalf("filter: %v", err)
	}
	sort.Strings(missing)
	if got, want := fmt.Sprint(missing), "[c d]"; got != want {
		t.Errorf("FilterMissing = %v, want [c d]", missing)
	}

	// Empty input -> empty (not nil) slice.
	got, err := store.FilterMissing(ctx, nil)
	if err != nil {
		t.Fatalf("filter empty: %v", err)
	}
	if got == nil {
		t.Error("FilterMissing(nil) returned nil; want empty slice")
	}
	if len(got) != 0 {
		t.Errorf("FilterMissing(nil) = %v, want []", got)
	}

	// Duplicate input ids in the missing set should appear only once.
	dupMissing, err := store.FilterMissing(ctx, []string{"x", "x", "x"})
	if err != nil {
		t.Fatalf("filter dup: %v", err)
	}
	if got, want := fmt.Sprint(dupMissing), "[x]"; got != want {
		t.Errorf("FilterMissing(dup x) = %v, want [x]", dupMissing)
	}
}

// Persist + reload via a fresh store backed by the same dir/namespace.
// Mirrors the property the s09 pipeline relies on every restart.
func TestKVPersistAcrossRestart(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()

	first, err := NewJSONKVStore(dir, "persist")
	if err != nil {
		t.Fatalf("new store: %v", err)
	}
	if err := first.Upsert(ctx, map[string]map[string]any{
		"a": {"n": 1},
		"b": {"n": 2},
		"c": {"n": 3},
	}); err != nil {
		t.Fatalf("upsert: %v", err)
	}
	if err := first.Persist(ctx); err != nil {
		t.Fatalf("persist: %v", err)
	}

	// `.tmp` must NOT exist after a successful Persist.
	if _, err := os.Stat(first.filePath() + ".tmp"); !os.IsNotExist(err) {
		t.Errorf(".tmp file lingers after Persist; got err=%v (want IsNotExist)", err)
	}

	second, err := NewJSONKVStore(dir, "persist")
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	if got, want := second.Len(), 3; got != want {
		t.Fatalf("after reload, Len = %d, want %d", got, want)
	}
	for _, id := range []string{"a", "b", "c"} {
		rec, ok, err := second.Get(ctx, id)
		if err != nil || !ok {
			t.Errorf("Get(%q) = (%v, %v, %v); want (rec, true, nil)", id, rec, ok, err)
		}
	}
}

// Hammer the per-key locking with 100 goroutines writing 10 disjoint keys
// each.  Run with `-race` to catch data races.  Final count must be 1000.
func TestKVConcurrentUpsertSafe(t *testing.T) {
	ctx := context.Background()
	store, err := NewJSONKVStore(t.TempDir(), "concurrent")
	if err != nil {
		t.Fatalf("new store: %v", err)
	}

	const writers = 100
	const perWriter = 10
	var wg sync.WaitGroup
	wg.Add(writers)
	for w := 0; w < writers; w++ {
		w := w
		go func() {
			defer wg.Done()
			batch := make(map[string]map[string]any, perWriter)
			for i := 0; i < perWriter; i++ {
				key := fmt.Sprintf("w%03d-i%03d", w, i)
				batch[key] = map[string]any{"w": w, "i": i}
			}
			if err := store.Upsert(ctx, batch); err != nil {
				t.Errorf("upsert from goroutine %d: %v", w, err)
			}
		}()
	}
	wg.Wait()

	if got, want := store.Len(), writers*perWriter; got != want {
		t.Errorf("Len after concurrent Upsert = %d, want %d", got, want)
	}
}

// A bogus `.tmp` file must NOT bleed into a fresh store's load.  Only the
// canonical file is the source of truth at startup.
func TestKVAtomicWriteOnCrash(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()

	first, err := NewJSONKVStore(dir, "atomic")
	if err != nil {
		t.Fatalf("new store: %v", err)
	}
	if err := first.Upsert(ctx, map[string]map[string]any{
		"good": {"n": 1},
	}); err != nil {
		t.Fatalf("upsert: %v", err)
	}
	if err := first.Persist(ctx); err != nil {
		t.Fatalf("persist: %v", err)
	}

	// Simulate a crashed Persist that left a half-written .tmp behind.
	tmp := first.filePath() + ".tmp"
	if err := os.WriteFile(tmp, []byte(`{"poisoned":{"n":99}`), 0o644); err != nil {
		t.Fatalf("seed tmp: %v", err)
	}

	// A new store opening the same (dir, ns) must ignore the .tmp.
	second, err := NewJSONKVStore(dir, "atomic")
	if err != nil {
		t.Fatalf("reopen with stale .tmp: %v", err)
	}
	if got, want := second.Len(), 1; got != want {
		t.Fatalf("Len after stale-tmp reopen = %d, want %d", got, want)
	}
	if _, ok, _ := second.Get(ctx, "poisoned"); ok {
		t.Error("`poisoned` was loaded from .tmp; expected canonical-only load")
	}
	rec, ok, _ := second.Get(ctx, "good")
	if !ok {
		t.Fatal("canonical `good` record missing after reopen")
	}
	if got := fmt.Sprint(rec["n"]); got != "1" {
		t.Errorf("good.n = %q, want \"1\"", got)
	}

	// Sanity: confirm the canonical file is well-formed JSON, in case the
	// reader wants to inspect it manually.
	canon, err := os.ReadFile(first.filePath())
	if err != nil {
		t.Fatalf("read canonical: %v", err)
	}
	var parsed map[string]map[string]any
	if err := json.Unmarshal(canon, &parsed); err != nil {
		t.Errorf("canonical file is malformed JSON: %v", err)
	}

	// Cleanup the orphan tmp so subsequent test runs don't pile on.
	_ = os.Remove(tmp)
	_ = filepath.Clean(dir) // no-op; just keeps the import live for any future use
}

// Delete + Get must surface a clean miss.
func TestKVDeleteRemovesKey(t *testing.T) {
	ctx := context.Background()
	store, err := NewJSONKVStore(t.TempDir(), "delete")
	if err != nil {
		t.Fatalf("new store: %v", err)
	}

	if err := store.Upsert(ctx, map[string]map[string]any{
		"k1": {"v": 1},
		"k2": {"v": 2},
	}); err != nil {
		t.Fatalf("upsert: %v", err)
	}
	if err := store.Delete(ctx, []string{"k1"}); err != nil {
		t.Fatalf("delete: %v", err)
	}

	if _, ok, _ := store.Get(ctx, "k1"); ok {
		t.Error("Get(k1) hit after Delete; want miss")
	}
	if _, ok, _ := store.Get(ctx, "k2"); !ok {
		t.Error("Get(k2) missed after deleting only k1")
	}

	// Deleting a non-existent id must be a silent no-op.
	if err := store.Delete(ctx, []string{"never-existed"}); err != nil {
		t.Errorf("Delete(missing) = %v, want nil", err)
	}
}
