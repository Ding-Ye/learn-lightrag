package main

// Chunk mirrors upstream TextChunkSchema (lightrag/base.py:71). In s01 we
// keep the shape but skip the real tiktoken count — s04 fills `Tokens` in
// for real with cl100k_base.

import "strings"

// Chunk is the canonical s01..s11 chunk struct.
type Chunk struct {
	ContentDocID    string // parent doc ID
	Content         string
	Tokens          int // s01: rune count stub; s04 replaces with tiktoken
	ChunkOrderIndex int
}

// ChunkByNewlines splits `text` into chunks. Strategy:
//
//  1. Split on "\n\n" to get paragraph blocks. This mirrors the natural
//     boundary that real-world documents already encode.
//  2. For any paragraph longer than maxRunes runes, hard-cut into
//     maxRunes-rune slices. Real tokenizer-aware chunking comes in s04.
//
// Empty / whitespace-only paragraphs are skipped so chunk indices stay dense.
func ChunkByNewlines(docID, text string, maxRunes int) []Chunk {
	if maxRunes <= 0 {
		maxRunes = 1200 // mirror upstream chunk_token_size default
	}
	var out []Chunk
	idx := 0
	for _, para := range strings.Split(text, "\n\n") {
		para = strings.TrimSpace(para)
		if para == "" {
			continue
		}
		runes := []rune(para)
		for start := 0; start < len(runes); start += maxRunes {
			end := start + maxRunes
			if end > len(runes) {
				end = len(runes)
			}
			body := string(runes[start:end])
			out = append(out, Chunk{
				ContentDocID:    docID,
				Content:         body,
				Tokens:          end - start, // rune count stub
				ChunkOrderIndex: idx,
			})
			idx++
		}
	}
	return out
}
