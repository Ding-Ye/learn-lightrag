package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
)

// CLI demo for s04 chunking.
//
//	go run . [-doc path] [-size 1200] [-overlap 100] [-tokenizer tiktoken|ws]
//
// Reads a text file, runs ChunkByTokenSize, and prints one line per
// chunk: index, token count, and the first 80 characters of content
// for a quick sanity scan.  The `-tokenizer ws` switch lets you run
// fully offline with no network — handy if the tiktoken merges file
// hasn't been cached yet under ~/.tiktoken/.
func main() {
	var (
		docPath      string
		chunkSize    int
		overlapSize  int
		tokenizerSel string
		splitByChar  bool
	)
	flag.StringVar(&docPath, "doc", "testdata/sample.txt", "path to the text document to chunk")
	flag.IntVar(&chunkSize, "size", 1200, "chunk size in tokens")
	flag.IntVar(&overlapSize, "overlap", 100, "overlap size in tokens")
	flag.StringVar(&tokenizerSel, "tokenizer", "tiktoken", "tokenizer to use: 'tiktoken' (cl100k_base) or 'ws' (whitespace fallback)")
	flag.BoolVar(&splitByChar, "split-by-character", false, "use the rune-based fallback path instead of token-window")
	flag.Parse()

	if err := run(docPath, chunkSize, overlapSize, tokenizerSel, splitByChar); err != nil {
		log.Fatalf("s04 demo failed: %v", err)
	}
}

func run(docPath string, chunkSize, overlapSize int, tokenizerSel string, splitByChar bool) error {
	content, err := os.ReadFile(docPath)
	if err != nil {
		return fmt.Errorf("read %s: %w", docPath, err)
	}
	text := string(content)

	var tokenizer Tokenizer
	switch tokenizerSel {
	case "tiktoken":
		tk, err := NewTiktokenTokenizer()
		if err != nil {
			return fmt.Errorf("init tiktoken: %w (tip: rerun with -tokenizer ws to skip the network)", err)
		}
		tokenizer = tk
	case "ws":
		tokenizer = NewWhitespaceTokenizer()
	default:
		return fmt.Errorf("unknown tokenizer %q (want 'tiktoken' or 'ws')", tokenizerSel)
	}

	// docID is just the filename for the demo — s03 supplies the real
	// MD5 content-address in the integrated pipeline.
	docID := strings.TrimSuffix(strings.ReplaceAll(docPath, "/", "_"), ".txt")

	chunks, err := ChunkByTokenSize(docID, text, tokenizer, chunkSize, overlapSize, splitByChar)
	if err != nil {
		return err
	}

	fmt.Printf("== input ==\n  path      = %s\n  bytes     = %d\n  tokenizer = %s\n  chunkSize = %d  overlap = %d\n  totalToks = %d\n\n",
		docPath, len(content), tokenizerSel, chunkSize, overlapSize, len(tokenizer.Encode(text)))
	fmt.Printf("== %d chunks ==\n", len(chunks))
	for _, c := range chunks {
		preview := c.Content
		if len(preview) > 80 {
			preview = preview[:80] + "..."
		}
		preview = strings.ReplaceAll(preview, "\n", " ")
		fmt.Printf("  [%02d] tokens=%4d  %s\n", c.ChunkOrderIndex, c.Tokens, preview)
	}
	return nil
}
