package main

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
	"time"
)

// fixedClock returns a deterministic timestamp so persist/load round-trips are
// byte-stable in tests.  Real callers leave Clock at its time.Now default.
func fixedClock() time.Time {
	return time.Date(2026, 5, 9, 12, 0, 0, 0, time.UTC)
}

func newTestStore(t *testing.T) *DocStatusStore {
	t.Helper()
	path := filepath.Join(t.TempDir(), "doc_status.json")
	s, err := NewDocStatusStore(path)
	if err != nil {
		t.Fatalf("NewDocStatusStore: %v", err)
	}
	s.Clock = fixedClock
	return s
}

// (1) Happy path — every legal edge fires in order; final state is PROCESSED
// and ChunksList round-trips.
func TestStatusTransitionsValid(t *testing.T) {
	ctx := context.Background()
	s := newTestStore(t)

	docID := MD5DocID("hello world")
	rec, fresh, err := s.Enqueue(ctx, docID, "hello world", "test.txt")
	if err != nil {
		t.Fatalf("Enqueue: %v", err)
	}
	if !fresh {
		t.Fatal("expected fresh=true on first enqueue")
	}
	if rec.Status != DocStatusPending {
		t.Errorf("after Enqueue: status=%s, want PENDING", rec.Status)
	}

	if err := s.MarkProcessing(ctx, docID); err != nil {
		t.Fatalf("MarkProcessing: %v", err)
	}
	chunks := []string{docID + "::chunk-0", docID + "::chunk-1"}
	if err := s.MarkProcessed(ctx, docID, chunks); err != nil {
		t.Fatalf("MarkProcessed: %v", err)
	}

	got, ok, err := s.Get(ctx, docID)
	if err != nil || !ok {
		t.Fatalf("Get: ok=%v err=%v", ok, err)
	}
	if got.Status != DocStatusProcessed {
		t.Errorf("final status=%s, want PROCESSED", got.Status)
	}
	if got.ChunksCount != 2 {
		t.Errorf("ChunksCount=%d, want 2", got.ChunksCount)
	}
	if len(got.ChunksList) != 2 || got.ChunksList[0] != chunks[0] {
		t.Errorf("ChunksList round-trip mismatch: got %v", got.ChunksList)
	}
}

// (2) PROCESSED is terminal — no edge to PROCESSING.
func TestStatusInvalidTransitionRejected(t *testing.T) {
	ctx := context.Background()
	s := newTestStore(t)

	docID := MD5DocID("doc-2")
	if _, _, err := s.Enqueue(ctx, docID, "doc-2", ""); err != nil {
		t.Fatal(err)
	}
	if err := s.MarkProcessing(ctx, docID); err != nil {
		t.Fatal(err)
	}
	if err := s.MarkProcessed(ctx, docID, []string{"chunk-0"}); err != nil {
		t.Fatal(err)
	}

	// Now try the forbidden edge: PROCESSED -> PROCESSING via internal helper.
	err := s.transition(ctx, docID, DocStatusProcessing, nil)
	if err == nil {
		t.Fatal("expected ErrInvalidTransition for PROCESSED -> PROCESSING; got nil")
	}
	if !errors.Is(err, ErrInvalidTransition) {
		t.Errorf("error chain doesn't include ErrInvalidTransition: %v", err)
	}
	var te *transitionError
	if !errors.As(err, &te) {
		t.Errorf("error not *transitionError: %v", err)
	} else if te.From != DocStatusProcessed || te.To != DocStatusProcessing {
		t.Errorf("transitionError From/To = %s/%s, want processed/processing", te.From, te.To)
	}
}

// (3) Persist, drop the in-memory state, Load, assert round-trip.
func TestStatusPersistAcrossRestart(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "doc_status.json")

	first, err := NewDocStatusStore(path)
	if err != nil {
		t.Fatalf("NewDocStatusStore: %v", err)
	}
	first.Clock = fixedClock

	docs := []struct {
		content string
		fp      string
	}{
		{"alpha document body", "alpha.txt"},
		{"beta body of text", "beta.txt"},
		{"gamma is the third", "gamma.txt"},
	}
	ids := make([]string, len(docs))
	for i, d := range docs {
		ids[i] = MD5DocID(d.content)
		if _, _, err := first.Enqueue(ctx, ids[i], d.content, d.fp); err != nil {
			t.Fatalf("enqueue %d: %v", i, err)
		}
	}
	// Mark one PROCESSED so we can verify status round-trips, not just IDs.
	if err := first.MarkProcessing(ctx, ids[1]); err != nil {
		t.Fatal(err)
	}
	if err := first.MarkProcessed(ctx, ids[1], []string{"c0", "c1"}); err != nil {
		t.Fatal(err)
	}
	if err := first.Persist(ctx); err != nil {
		t.Fatalf("Persist: %v", err)
	}

	// Construct a brand-new store at the same path; constructor calls Load.
	second, err := NewDocStatusStore(path)
	if err != nil {
		t.Fatalf("re-open: %v", err)
	}
	for i, id := range ids {
		got, ok, err := second.Get(ctx, id)
		if err != nil || !ok {
			t.Fatalf("after reload, doc %d (%s): ok=%v err=%v", i, id, ok, err)
		}
		if got.FilePath != docs[i].fp {
			t.Errorf("doc %d FilePath=%q want %q", i, got.FilePath, docs[i].fp)
		}
	}
	got, _, _ := second.Get(ctx, ids[1])
	if got.Status != DocStatusProcessed {
		t.Errorf("doc 1 status after reload = %s, want PROCESSED", got.Status)
	}
	if len(got.ChunksList) != 2 {
		t.Errorf("doc 1 ChunksList after reload = %v, want 2 entries", got.ChunksList)
	}
}

// (4) Re-Enqueue the same content returns (existing, false, nil) AND the
// docID is byte-equal between calls — content addressing in action.
func TestDuplicateInsertIsNoop(t *testing.T) {
	ctx := context.Background()
	s := newTestStore(t)

	content := "the same paragraph, repeated"
	id1 := MD5DocID(content)
	id2 := MD5DocID(content)
	if id1 != id2 {
		t.Fatalf("MD5DocID not deterministic: %s vs %s", id1, id2)
	}

	rec1, fresh1, err := s.Enqueue(ctx, id1, content, "first.txt")
	if err != nil {
		t.Fatal(err)
	}
	if !fresh1 {
		t.Fatal("first call should be fresh")
	}

	// Mutate the first record's status so we can detect overwrite if it happens.
	if err := s.MarkProcessing(ctx, id1); err != nil {
		t.Fatal(err)
	}

	rec2, fresh2, err := s.Enqueue(ctx, id2, content, "second.txt")
	if err != nil {
		t.Fatal(err)
	}
	if fresh2 {
		t.Fatal("second call should NOT be fresh (dedup)")
	}
	if rec2.Status != DocStatusProcessing {
		t.Errorf("second Enqueue clobbered status: got %s, want PROCESSING (preserved)", rec2.Status)
	}
	if rec2.FilePath != rec1.FilePath {
		t.Errorf("second Enqueue clobbered FilePath: got %q, want %q", rec2.FilePath, rec1.FilePath)
	}
}

// (5) MarkFailed preserves the caller's error message verbatim.
func TestFailedDocPreservesErrorMsg(t *testing.T) {
	ctx := context.Background()
	s := newTestStore(t)

	docID := MD5DocID("doc-5")
	if _, _, err := s.Enqueue(ctx, docID, "doc-5", ""); err != nil {
		t.Fatal(err)
	}
	if err := s.MarkProcessing(ctx, docID); err != nil {
		t.Fatal(err)
	}
	if err := s.MarkFailed(ctx, docID, "oh no"); err != nil {
		t.Fatalf("MarkFailed: %v", err)
	}

	got, ok, err := s.Get(ctx, docID)
	if err != nil || !ok {
		t.Fatalf("Get: ok=%v err=%v", ok, err)
	}
	if got.Status != DocStatusFailed {
		t.Errorf("status=%s, want FAILED", got.Status)
	}
	if got.ErrorMsg != "oh no" {
		t.Errorf("ErrorMsg=%q, want \"oh no\"", got.ErrorMsg)
	}

	// And ListByStatus surfaces it for the resume scan.
	failed, err := s.ListByStatus(ctx, DocStatusFailed)
	if err != nil {
		t.Fatal(err)
	}
	if len(failed) != 1 || failed[0].DocID != docID {
		t.Errorf("ListByStatus(FAILED) = %v, want exactly [%s]", failed, docID)
	}
}
