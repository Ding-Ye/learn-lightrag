package main

import (
	"fmt"
	"strings"

	tiktoken "github.com/pkoukk/tiktoken-go"
)

// Tokenizer mirrors upstream's `Tokenizer` protocol from
// `lightrag/utils.py` — `chunking_by_token_size` only ever calls
// `.encode(text)` and `.decode(tokens)`, so this two-method interface
// is the entire surface s04 needs.
//
// Two implementations live here:
//
//   - TiktokenTokenizer wraps github.com/pkoukk/tiktoken-go's cl100k_base
//     encoder (the same BPE OpenAI uses for `text-embedding-3-small`,
//     `gpt-4o-mini`, etc.).  This is the one production code wants.
//   - WhitespaceTokenizer is a stdlib-only fallback used by tests + by
//     the demo's `--tokenizer ws` flag for offline runs.  Encoding is
//     `strings.Fields`; decoding joins with a single space.  Token
//     counts match upstream's "one token per whitespace-delimited word"
//     intuition closely enough for the chunking algorithm to behave
//     identically — only the absolute size differs.
type Tokenizer interface {
	Encode(text string) []int
	Decode(tokens []int) string
}

// TiktokenTokenizer wraps the cl100k_base encoder.  The first call to
// `NewTiktokenTokenizer` triggers a ~1.6 MB download of the merges file
// from the tiktoken-go vendor URL on cache miss; subsequent calls hit
// the on-disk cache at `~/.tiktoken/`.  Document this in the README so
// CI failures on a flaky network are easy to diagnose.
type TiktokenTokenizer struct {
	enc *tiktoken.Tiktoken
}

// NewTiktokenTokenizer returns a TiktokenTokenizer over cl100k_base, or
// the underlying tiktoken-go error if the BPE merges file can't be
// loaded.  Most callers don't need to wrap the error — fail fast on
// startup is the safe choice when chunking is the very first stage of
// the ingest pipeline.
func NewTiktokenTokenizer() (*TiktokenTokenizer, error) {
	enc, err := tiktoken.GetEncoding("cl100k_base")
	if err != nil {
		return nil, fmt.Errorf("s04: load cl100k_base: %w", err)
	}
	return &TiktokenTokenizer{enc: enc}, nil
}

// Encode splits text into BPE token IDs.  No special-token handling —
// upstream `chunking_by_token_size` doesn't pass any either.
func (t *TiktokenTokenizer) Encode(text string) []int {
	return t.enc.Encode(text, nil, nil)
}

// Decode reverses Encode.  Round-trip is byte-stable for any text that
// originally came from Encode; arbitrary token slices may produce text
// that re-encodes differently (a known BPE property — not a bug).
func (t *TiktokenTokenizer) Decode(tokens []int) string {
	return t.enc.Decode(tokens)
}

// WhitespaceTokenizer encodes via `strings.Fields` (any Unicode
// whitespace as separator) and decodes by joining with a single space.
// Round-tripping `Decode(Encode(s))` is NOT byte-stable for inputs with
// runs of whitespace — but for the chunking-algorithm tests we only
// care that:
//
//  1. token-count comparisons (`len(Encode(...))`) are deterministic;
//  2. `Decode(Encode(s))` is non-empty whenever `s` has any non-space
//     content;
//  3. concatenating chunks back together yields a re-tokenizable text.
//
// All three hold here.
type WhitespaceTokenizer struct {
	// vocab maps token-string -> token-id; we mint IDs on first sight
	// so Encode is stable across one process lifetime.  Tests share a
	// single tokenizer instance per test, so this is enough.
	vocab map[string]int
	rev   []string
}

// NewWhitespaceTokenizer returns a fresh whitespace tokenizer.  Token
// IDs start at 0 and grow with vocabulary; this is intentionally
// process-local — there is no on-disk vocab file to load.
func NewWhitespaceTokenizer() *WhitespaceTokenizer {
	return &WhitespaceTokenizer{
		vocab: make(map[string]int),
		rev:   make([]string, 0, 64),
	}
}

// Encode splits on Unicode whitespace and assigns/looks-up an int ID
// for each unique word.  Order is preserved.
func (w *WhitespaceTokenizer) Encode(text string) []int {
	fields := strings.Fields(text)
	out := make([]int, len(fields))
	for i, f := range fields {
		id, ok := w.vocab[f]
		if !ok {
			id = len(w.rev)
			w.vocab[f] = id
			w.rev = append(w.rev, f)
		}
		out[i] = id
	}
	return out
}

// Decode joins token IDs back into a single space-separated string.
// Unknown IDs (e.g. tokens minted by another tokenizer) become empty
// strings; chunking's caller never crosses tokenizer instances so this
// edge case is theoretical.
func (w *WhitespaceTokenizer) Decode(tokens []int) string {
	parts := make([]string, len(tokens))
	for i, id := range tokens {
		if id >= 0 && id < len(w.rev) {
			parts[i] = w.rev[id]
		}
	}
	return strings.Join(parts, " ")
}
