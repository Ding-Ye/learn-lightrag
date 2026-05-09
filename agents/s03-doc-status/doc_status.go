// Package main — s03 (doc-status) declares the document state machine that
// mirrors upstream `lightrag/base.py:662-697`.  The shape is intentionally
// re-typed verbatim per the curriculum's "no cross-session imports" rule;
// every later session that needs DocProcessingStatus re-types it the same way.
package main

import (
	"errors"
	"fmt"
	"time"
)

// DocStatus is the four-state ingestion lifecycle a document moves through.
// Mirrors `class DocStatus(str, Enum)` at lightrag/base.py:662 — we drop
// upstream's PREPROCESSED state because s03 doesn't teach multimodal.
type DocStatus string

const (
	DocStatusPending    DocStatus = "pending"
	DocStatusProcessing DocStatus = "processing"
	DocStatusProcessed  DocStatus = "processed"
	DocStatusFailed     DocStatus = "failed"
)

// DocProcessingStatus is the per-doc state record persisted by DocStatusStore.
// Mirrors the dataclass at lightrag/base.py:672-706.  Field names are exported
// (Go convention) and JSON-tagged so the on-disk file matches upstream's
// `kv_store_doc_status.json` snake_case convention.
type DocProcessingStatus struct {
	DocID          string         `json:"doc_id"`
	ContentSummary string         `json:"content_summary"`
	ContentLength  int            `json:"content_length"`
	FilePath       string         `json:"file_path"`
	Status         DocStatus      `json:"status"`
	CreatedAt      time.Time      `json:"created_at"`
	UpdatedAt      time.Time      `json:"updated_at"`
	TrackID        string         `json:"track_id,omitempty"`
	ChunksCount    int            `json:"chunks_count"`
	ChunksList     []string       `json:"chunks_list"`
	ErrorMsg       string         `json:"error_msg,omitempty"`
	Metadata       map[string]any `json:"metadata,omitempty"`
}

// ErrInvalidTransition is returned by Mark* methods when the requested edge
// is not in the state machine.  Callers can use errors.Is / errors.As to
// distinguish it from I/O errors during persistence.
var ErrInvalidTransition = errors.New("invalid doc-status transition")

// transitionError carries the From/To context so callers can log it; the
// sentinel above is preserved via Unwrap() for errors.Is().
type transitionError struct {
	DocID string
	From  DocStatus
	To    DocStatus
}

func (e *transitionError) Error() string {
	return fmt.Sprintf("doc-status: %s: cannot transition %s -> %s", e.DocID, e.From, e.To)
}

func (e *transitionError) Unwrap() error { return ErrInvalidTransition }

// IsValidTransition encodes the upstream state machine as a 2-D table.
// Allowed edges:
//
//	PENDING    -> PROCESSING | FAILED
//	PROCESSING -> PROCESSED  | FAILED
//	PROCESSED  -> (terminal — no outgoing edges)
//	FAILED     -> PROCESSING (retry; matches resume-on-failure semantics)
//
// Self-loops (e.g. PROCESSING -> PROCESSING) are rejected to surface caller
// bugs (a worker double-marking the same doc).  The retry edge from FAILED
// back to PROCESSING is what makes the upstream "resume-on-failure" feature
// possible without duplicating chunks.
func IsValidTransition(from, to DocStatus) bool {
	switch from {
	case DocStatusPending:
		return to == DocStatusProcessing || to == DocStatusFailed
	case DocStatusProcessing:
		return to == DocStatusProcessed || to == DocStatusFailed
	case DocStatusFailed:
		return to == DocStatusProcessing
	case DocStatusProcessed:
		return false // terminal
	default:
		return false
	}
}
