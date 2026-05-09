package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
)

// CLI demo: ingest a text file twice (proving idempotency), then synthetically
// fail it on a third attempt to demonstrate the resume-on-failure scan.
//
// Run with `go run . [-doc path]`.  Default `-doc testdata/sample.txt`.
func main() {
	var docPath string
	flag.StringVar(&docPath, "doc", "testdata/sample.txt", "path to the document to ingest")
	flag.Parse()

	if err := run(docPath); err != nil {
		log.Fatalf("s03 demo failed: %v", err)
	}
}

func run(docPath string) error {
	ctx := context.Background()

	store, err := NewDocStatusStore("")
	if err != nil {
		return fmt.Errorf("new store: %w", err)
	}

	content, err := os.ReadFile(docPath)
	if err != nil {
		return fmt.Errorf("read %s: %w", docPath, err)
	}
	docID := MD5DocID(string(content))
	fmt.Printf("== input doc ==\n  path     = %s\n  bytes    = %d\n  doc_id   = %s\n\n", docPath, len(content), docID)

	// --- Round 1: first ingest, full happy path PENDING -> PROCESSING -> PROCESSED ---
	fmt.Println("== round 1: fresh ingest ==")
	rec, fresh, err := store.Enqueue(ctx, docID, string(content), docPath)
	if err != nil {
		return err
	}
	logTransition("Enqueue", "(none)", rec.Status, fresh, rec)

	if err := store.MarkProcessing(ctx, docID); err != nil {
		return err
	}
	logTransition("MarkProcessing", DocStatusPending, DocStatusProcessing, false, mustGet(ctx, store, docID))

	chunkIDs := []string{docID + "::chunk-0", docID + "::chunk-1", docID + "::chunk-2"}
	if err := store.MarkProcessed(ctx, docID, chunkIDs); err != nil {
		return err
	}
	logTransition("MarkProcessed", DocStatusProcessing, DocStatusProcessed, false, mustGet(ctx, store, docID))

	// --- Round 2: re-ingest the same content; idempotency proof ---
	fmt.Println("\n== round 2: re-ingest same content (dedup) ==")
	_, fresh2, err := store.Enqueue(ctx, docID, string(content), docPath)
	if err != nil {
		return err
	}
	if fresh2 {
		return fmt.Errorf("expected fresh=false on duplicate ingest, got true")
	}
	fmt.Printf("  Enqueue returned fresh=false (no-op) — content already PROCESSED, no double-chunking.\n")

	// --- Round 3: force a failure on a different doc, then list FAILED for resume ---
	fmt.Println("\n== round 3: simulate failure on a second doc ==")
	failContent := "(simulated) document that will fail mid-pipeline"
	failID := MD5DocID(failContent)
	if _, _, err := store.Enqueue(ctx, failID, failContent, "<synthetic>"); err != nil {
		return err
	}
	if err := store.MarkProcessing(ctx, failID); err != nil {
		return err
	}
	if err := store.MarkFailed(ctx, failID, "synthetic: provider rate-limit"); err != nil {
		return err
	}
	fmt.Printf("  doc %s → FAILED (err=%q)\n", failID[:8], "synthetic: provider rate-limit")

	// Demonstrate that a forbidden transition surfaces ErrInvalidTransition.
	if err := store.MarkProcessed(ctx, failID, nil); err != nil {
		if errors.Is(err, ErrInvalidTransition) {
			fmt.Printf("  attempting FAILED -> PROCESSED rejected as expected: %v\n", err)
		} else {
			return fmt.Errorf("unexpected error type: %w", err)
		}
	}

	// --- The resume-on-failure scan ---
	fmt.Println("\n== resume-on-failure scan ==")
	failed, err := store.ListByStatus(ctx, DocStatusFailed)
	if err != nil {
		return err
	}
	fmt.Printf("  ListByStatus(FAILED) found %d doc(s) needing retry:\n", len(failed))
	for _, d := range failed {
		fmt.Printf("    - %s  err=%q  updated=%s\n", d.DocID[:8], d.ErrorMsg, d.UpdatedAt.Format("15:04:05"))
	}

	// Persist for the next process to load.
	if err := store.Persist(ctx); err != nil {
		return err
	}
	fmt.Printf("\n  persisted %d records to ./lightrag-data/doc_status.json\n", 2)
	return nil
}

func mustGet(ctx context.Context, s *DocStatusStore, id string) *DocProcessingStatus {
	rec, _, _ := s.Get(ctx, id)
	return rec
}

func logTransition(op string, from any, to DocStatus, fresh bool, rec *DocProcessingStatus) {
	suffix := ""
	if op == "Enqueue" {
		if fresh {
			suffix = " (fresh)"
		} else {
			suffix = " (dedup hit)"
		}
	}
	chunks := 0
	if rec != nil {
		chunks = rec.ChunksCount
	}
	fmt.Printf("  %-15s %v -> %s%s  chunks=%d\n", op, from, to, suffix, chunks)
}
