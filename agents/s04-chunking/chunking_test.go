package main

import (
	"errors"
	"strings"
	"testing"
)

// All tests in this file use WhitespaceTokenizer so they run fully
// offline (no tiktoken merges-file download).  TiktokenTokenizer's
// Tokenizer-interface satisfaction is checked at compile time via the
// `_ Tokenizer = (*TiktokenTokenizer)(nil)` assertion in chunking.go,
// and again here in TestTokenizerInterfaceContract.

// repeatWords returns text containing exactly `n` whitespace-tokens so
// the WhitespaceTokenizer's Encode produces an `n`-element slice.
func repeatWords(n int) string {
	parts := make([]string, n)
	for i := 0; i < n; i++ {
		parts[i] = "word"
	}
	return strings.Join(parts, " ")
}

// (1) Every chunk must have Tokens <= chunkTokenSize.
func TestChunkRespectsTokenLimit(t *testing.T) {
	tk := NewWhitespaceTokenizer()
	const chunkSize, overlap = 50, 10
	text := repeatWords(231) // not a multiple of step
	chunks, err := ChunkByTokenSize("doc-1", text, tk, chunkSize, overlap, false)
	if err != nil {
		t.Fatalf("ChunkByTokenSize: %v", err)
	}
	if len(chunks) == 0 {
		t.Fatal("expected at least one chunk")
	}
	for i, c := range chunks {
		if c.Tokens > chunkSize {
			t.Errorf("chunk[%d] Tokens=%d > chunkSize=%d", i, c.Tokens, chunkSize)
		}
		if c.Tokens <= 0 {
			t.Errorf("chunk[%d] Tokens=%d, want > 0", i, c.Tokens)
		}
	}
}

// (2) Adjacent chunks share `overlap` tokens.  We verify by checking
// that the last K WS-tokens of chunk[i] equal the first K WS-tokens of
// chunk[i+1] for K = overlap.
func TestChunkOverlapPreserved(t *testing.T) {
	tk := NewWhitespaceTokenizer()
	const chunkSize, overlap = 30, 5
	text := repeatWords(120)
	chunks, err := ChunkByTokenSize("doc-2", text, tk, chunkSize, overlap, false)
	if err != nil {
		t.Fatalf("ChunkByTokenSize: %v", err)
	}
	if len(chunks) < 2 {
		t.Fatalf("need >= 2 chunks to check overlap, got %d", len(chunks))
	}

	// Build a unique-token text so we can detect mis-alignment.
	parts := make([]string, 120)
	for i := range parts {
		parts[i] = wordN(i)
	}
	uniqueText := strings.Join(parts, " ")
	uniqueTokenizer := NewWhitespaceTokenizer()
	uniqueChunks, err := ChunkByTokenSize("doc-2u", uniqueText, uniqueTokenizer, chunkSize, overlap, false)
	if err != nil {
		t.Fatalf("ChunkByTokenSize unique: %v", err)
	}
	if len(uniqueChunks) < 2 {
		t.Fatalf("unique-text need >= 2 chunks, got %d", len(uniqueChunks))
	}

	for i := 0; i+1 < len(uniqueChunks); i++ {
		// All but the last chunk should be exactly chunkSize tokens.
		if uniqueChunks[i].Tokens != chunkSize {
			continue // last chunk in slice may be short — but we filter via i+1 < len anyway
		}
		left := strings.Fields(uniqueChunks[i].Content)
		right := strings.Fields(uniqueChunks[i+1].Content)
		if len(left) < overlap || len(right) < overlap {
			t.Fatalf("chunk too short to slice overlap: left=%d right=%d", len(left), len(right))
		}
		tail := left[len(left)-overlap:]
		head := right[:overlap]
		for j := range tail {
			if tail[j] != head[j] {
				t.Errorf("overlap mismatch between chunks %d/%d at offset %d: %q vs %q",
					i, i+1, j, tail[j], head[j])
			}
		}
	}
}

// (3) ChunkOrderIndex is 0..N-1 contiguously.
func TestChunkOrderIndexMonotonic(t *testing.T) {
	tk := NewWhitespaceTokenizer()
	chunks, err := ChunkByTokenSize("doc-3", repeatWords(500), tk, 60, 10, false)
	if err != nil {
		t.Fatalf("ChunkByTokenSize: %v", err)
	}
	if len(chunks) == 0 {
		t.Fatal("expected chunks")
	}
	for i, c := range chunks {
		if c.ChunkOrderIndex != i {
			t.Errorf("chunks[%d].ChunkOrderIndex = %d, want %d", i, c.ChunkOrderIndex, i)
		}
	}
}

// (4) splitByCharacter=true splits a long no-whitespace string into
// multiple chunks (the WhitespaceTokenizer would have produced exactly
// 1 token for it without the rune-fallback path).
func TestChunkSplitByCharacterFallback(t *testing.T) {
	tk := NewWhitespaceTokenizer()
	const chunkSize, overlap = 20, 5
	// Build a 1000-rune blob with no whitespace.  WhitespaceTokenizer's
	// Encode would give us a single token; the rune fallback splits on
	// rune count instead, so we expect multiple chunks.
	blob := strings.Repeat("A", 1000)
	chunks, err := ChunkByTokenSize("doc-4", blob, tk, chunkSize, overlap, true)
	if err != nil {
		t.Fatalf("ChunkByTokenSize splitByCharacter: %v", err)
	}
	if len(chunks) < 2 {
		t.Errorf("expected >= 2 chunks under splitByCharacter, got %d", len(chunks))
	}
	for i, c := range chunks {
		if c.Content == "" {
			t.Errorf("chunks[%d].Content empty", i)
		}
	}
}

// (5) Empty input returns 0 chunks AND ErrEmptyText so the caller can
// log a reason — picking "no chunks + error" matches the brief.
func TestChunkEmptyTextReturnsEmpty(t *testing.T) {
	tk := NewWhitespaceTokenizer()
	chunks, err := ChunkByTokenSize("doc-5", "", tk, 100, 20, false)
	if !errors.Is(err, ErrEmptyText) {
		t.Errorf("err = %v, want ErrEmptyText", err)
	}
	if len(chunks) != 0 {
		t.Errorf("len(chunks) = %d, want 0", len(chunks))
	}

	// Whitespace-only is also empty.
	chunks, err = ChunkByTokenSize("doc-5b", "   \t\n  ", tk, 100, 20, false)
	if !errors.Is(err, ErrEmptyText) {
		t.Errorf("whitespace-only: err = %v, want ErrEmptyText", err)
	}
	if len(chunks) != 0 {
		t.Errorf("whitespace-only: len(chunks) = %d, want 0", len(chunks))
	}
}

// (6) Boundary behavior: exactly chunkSize tokens => 1 chunk;
// chunkSize+1 tokens => 2 chunks (the second is the overlap tail + 1
// new token).
func TestChunkExactlyAtBoundary(t *testing.T) {
	tk := NewWhitespaceTokenizer()
	const chunkSize, overlap = 40, 5

	exact := repeatWords(chunkSize)
	chunksA, err := ChunkByTokenSize("doc-6a", exact, tk, chunkSize, overlap, false)
	if err != nil {
		t.Fatalf("exact: %v", err)
	}
	if len(chunksA) != 1 {
		t.Errorf("len(chunksA) = %d, want 1 for exactly-chunkSize input", len(chunksA))
	}
	if len(chunksA) == 1 && chunksA[0].Tokens != chunkSize {
		t.Errorf("chunksA[0].Tokens = %d, want %d", chunksA[0].Tokens, chunkSize)
	}

	overflow := repeatWords(chunkSize + 1)
	chunksB, err := ChunkByTokenSize("doc-6b", overflow, tk, chunkSize, overlap, false)
	if err != nil {
		t.Fatalf("overflow: %v", err)
	}
	if len(chunksB) != 2 {
		t.Errorf("len(chunksB) = %d, want 2 for chunkSize+1 input", len(chunksB))
	}
}

// Compile-time interface contract: every Tokenizer impl in this
// package satisfies the interface.  This catches future renames at
// build time, not at first use.
func TestTokenizerInterfaceContract(t *testing.T) {
	var _ Tokenizer = (*TiktokenTokenizer)(nil)
	var _ Tokenizer = (*WhitespaceTokenizer)(nil)

	// Smoke check: a fresh whitespace tokenizer round-trips a simple
	// sentence.  Tiktoken is excluded from the runtime check because it
	// requires a network/cache lookup.
	ws := NewWhitespaceTokenizer()
	in := "the quick brown fox"
	got := ws.Decode(ws.Encode(in))
	if got != in {
		t.Errorf("WS round-trip: got %q, want %q", got, in)
	}
}

// wordN is a unique-token generator so overlap tests can detect
// mis-alignment.  Keeping it at file scope (not inside a test) lets
// TestChunkOverlapPreserved use it without a closure.
func wordN(n int) string {
	const charset = "abcdefghijklmnopqrstuvwxyz"
	var out [4]byte
	for i := range out {
		out[i] = charset[(n+i*13)%len(charset)]
	}
	return string(out[:]) + intToStr(n)
}

func intToStr(n int) string {
	if n == 0 {
		return "0"
	}
	var digits [20]byte
	i := len(digits)
	for n > 0 {
		i--
		digits[i] = byte('0' + n%10)
		n /= 10
	}
	return string(digits[i:])
}
