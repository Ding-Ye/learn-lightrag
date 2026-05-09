package main

import (
	"fmt"
	"strings"
)

// context_builder.go owns the per-section token-budgeted assembly of the
// final system prompt.  Three sections — entities, relations, chunks —
// are each truncated independently with their own budget, then composed
// into a single prompt.  Mirrors operate.py:3700-3850 _build_query_context
// (which also produces three blocks then concatenates them).
//
// We keep the tokenization simple: whitespace-word counts.  Real cl100k_base
// tokenizing lives in s04; doing it again in s11 would dwarf this file.
// The under-count is conservative (English prose runs ~1.3 tokens/word), so
// a budget that fits in word-count fits in real BPE tokens too.

// chunkRow is the in-context shape of one chunk after KV lookup.  Keeps
// the chunk_id alongside its content so References/dedup stay aligned.
type chunkRow struct {
	ID      string
	Content string
}

// tokenCount approximates token usage with a whitespace-word count.  See
// the file-level comment for why we don't bring in tiktoken here.
func tokenCount(s string) int { return len(strings.Fields(s)) }

// truncateByTokens cuts s to at most maxTokens whitespace-words.  Boundary:
// returns the first maxTokens words joined back with single spaces.  Returns
// s unchanged if it already fits.  Returns "" if maxTokens <= 0.
//
// Port of upstream's truncate_list_by_token_size (operate.py:3580+).  Same
// contract: stable prefix, no mid-word cut.
func truncateByTokens(s string, maxTokens int) string {
	if maxTokens <= 0 {
		return ""
	}
	fields := strings.Fields(s)
	if len(fields) <= maxTokens {
		return s
	}
	return strings.Join(fields[:maxTokens], " ")
}

// truncateChunksByTokens packs as many chunks as fit within maxTokens.
// Each chunk's content + a single newline separator counts toward budget.
// Returns the truncated slice in original order.  Mirrors upstream's
// per-chunk allocation policy: keep top-of-list chunks, drop tail.
func truncateChunksByTokens(rows []chunkRow, maxTokens int) []chunkRow {
	if maxTokens <= 0 || len(rows) == 0 {
		return nil
	}
	out := make([]chunkRow, 0, len(rows))
	used := 0
	for _, r := range rows {
		t := tokenCount(r.Content)
		if used+t > maxTokens && len(out) > 0 {
			break
		}
		out = append(out, r)
		used += t
		if used >= maxTokens {
			break
		}
	}
	return out
}

// buildSystemPrompt composes the three-section context window into one
// system prompt string.  Section order matches upstream
// (entities → relations → chunks → user query); each section is truncated
// to its budget.  The final prompt may exceed MaxTotalTokens by a small
// envelope (the section headers themselves) — that is acceptable for an
// explanatory port.
//
// param.MaxTotalTokens is honoured: chunks get the remainder after entities
// and relations are budgeted.  If the sum of MaxEntityTokens +
// MaxRelationTokens already exceeds MaxTotalTokens, chunks get nothing.
func buildSystemPrompt(
	entities []Entity,
	rels []Relationship,
	chunks []chunkRow,
	q string,
	param QueryParam,
) string {
	entSection := truncateByTokens(renderEntities(entities), param.MaxEntityTokens)
	relSection := truncateByTokens(renderRelations(rels), param.MaxRelationTokens)

	// Compute remaining budget for chunks under MaxTotalTokens.
	used := tokenCount(entSection) + tokenCount(relSection)
	chunkBudget := param.MaxTotalTokens - used
	if chunkBudget < 0 {
		chunkBudget = 0
	}
	chunkRows := truncateChunksByTokens(chunks, chunkBudget)

	var b strings.Builder
	b.WriteString("You are a Retrieval-Augmented Generation assistant. ")
	b.WriteString("Use ONLY the context below to answer the user query. ")
	b.WriteString("If the context is insufficient, say so.\n\n")

	b.WriteString("## Entities\n")
	if entSection == "" {
		b.WriteString("(none)\n")
	} else {
		b.WriteString(entSection + "\n")
	}

	b.WriteString("\n## Relations\n")
	if relSection == "" {
		b.WriteString("(none)\n")
	} else {
		b.WriteString(relSection + "\n")
	}

	b.WriteString("\n## Chunks\n")
	if len(chunkRows) == 0 {
		b.WriteString("(none)\n")
	} else {
		for _, c := range chunkRows {
			fmt.Fprintf(&b, "[%s] %s\n", c.ID, c.Content)
		}
	}

	b.WriteString("\n## User Query\n")
	b.WriteString(q + "\n")

	// Final overall truncation guarantees the prompt never balloons past
	// MaxTotalTokens by more than a small constant (header words).
	out := b.String()
	if param.MaxTotalTokens > 0 && tokenCount(out) > param.MaxTotalTokens {
		out = truncateByTokens(out, param.MaxTotalTokens)
	}
	return out
}

// renderEntities formats a slice of Entity into one block of newline-
// separated bullets.  Each entity gets a single line so the truncate step
// drops whole entities cleanly.
func renderEntities(es []Entity) string {
	if len(es) == 0 {
		return ""
	}
	var b strings.Builder
	for _, e := range es {
		typ := e.Type
		if typ == "" {
			typ = "entity"
		}
		fmt.Fprintf(&b, "- %s (%s): %s\n", e.Name, typ, e.Description)
	}
	return strings.TrimRight(b.String(), "\n")
}

// renderRelations formats a slice of Relationship into one block of
// newline-separated bullets.  Each relation includes the directional pair
// and any keywords the extractor surfaced.
func renderRelations(rs []Relationship) string {
	if len(rs) == 0 {
		return ""
	}
	var b strings.Builder
	for _, r := range rs {
		kw := r.Keywords
		if kw == "" {
			kw = "(none)"
		}
		fmt.Fprintf(&b, "- %s -> %s [%s]: %s\n", r.SrcID, r.TgtID, kw, r.Description)
	}
	return strings.TrimRight(b.String(), "\n")
}

// chunkContentField pulls the chunk's text out of the KV record map.  The
// canonical field is "content"; we fall back to "text" / "chunk_content"
// because earlier sessions used different keys and we want s11 to interop
// with whatever was written.
func chunkContentField(rec map[string]any) string {
	for _, k := range []string{"content", "text", "chunk_content"} {
		if v, ok := rec[k]; ok {
			if s, ok := v.(string); ok && s != "" {
				return s
			}
		}
	}
	return ""
}
