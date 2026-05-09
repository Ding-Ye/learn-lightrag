package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// keywords.go owns the dual-level keyword extraction step.  Local mode uses
// only low-level keywords; global mode uses only high-level keywords; hybrid
// uses both.  Implementation:
//
//   1. Send a system prompt asking the LLM for STRICT JSON output with two
//      arrays: high_level_keywords and low_level_keywords.
//   2. Parse the response with parseKeywordJSON, which strips markdown code
//      fences if the LLM ignores rule (1) and wraps in ```json ... ```.
//   3. Return both lists.
//
// This is the s11 port of operate.py:3284 extract_keywords_only and the
// PROMPTS["keywords_extraction"] template at prompt.py:374.  The prompt
// constant below is a faithful summary, not verbatim — see
// upstream-readings/s11-query.py for the unsummarized excerpt.

// keywordExtractionSystem is the system prompt that instructs the LLM to
// emit a JSON object with two arrays.  Faithful to upstream
// PROMPTS["keywords_extraction"] (prompt.py:374-394) but compressed to a
// Go string constant for didactic clarity.  Trade-offs we kept:
//   - JSON output (not delimiter-based) — mode-router needs both lists
//     reliably, and JSON parsers are stricter than upstream's split-by-pipe.
//   - Empty-list contract for vague queries — lets caller short-circuit.
//   - Multi-word phrases preferred over single tokens — matches upstream.
const keywordExtractionSystem = `You are a keyword extraction agent for a Retrieval-Augmented Generation (RAG) system.

Given a user query, return a JSON object with exactly two fields:
  - "high_level_keywords": array of overarching themes / concepts / question types
  - "low_level_keywords":  array of specific entities, proper nouns, or concrete items

Rules:
  1. Output ONLY a single JSON object. No prose, no markdown fences, no comments.
  2. All keywords must be derivable from the query text.
  3. Prefer multi-word phrases when they represent one concept (e.g. "financial report", not "financial" + "report").
  4. For vague or trivial queries (e.g. "hi", "ok"), return both arrays empty.

Format:
  {"high_level_keywords": ["..."], "low_level_keywords": ["..."]}
`

// extractKeywords runs ONE LLM call to extract dual-level keywords from the
// query.  It returns (highLevel, lowLevel, error).  Empty lists are NOT an
// error — they signal a query the LLM judged too vague to retrieve on, and
// callers fall back to seed strategies (operate.py:3225-3232 in upstream).
func extractKeywords(ctx context.Context, p Provider, query string) ([]string, []string, error) {
	if p == nil {
		return nil, nil, fmt.Errorf("extractKeywords: nil Provider")
	}
	if strings.TrimSpace(query) == "" {
		return nil, nil, nil
	}
	resp, err := p.Complete(ctx, CompleteRequest{
		System: keywordExtractionSystem,
		Messages: []Message{
			{Role: "user", Content: "User Query: " + query + "\n\nOutput:"},
		},
		Temperature: 0.0,
	})
	if err != nil {
		return nil, nil, fmt.Errorf("extractKeywords: provider: %w", err)
	}
	hl, ll, err := parseKeywordJSON(resp.Text)
	if err != nil {
		return nil, nil, fmt.Errorf("extractKeywords: parse: %w (raw=%q)", err, resp.Text)
	}
	return hl, ll, nil
}

// keywordsPayload is the JSON shape the LLM is asked to emit.  Both fields
// are tagged to accept either snake_case (the spec) or camelCase (a common
// LLM drift) without an extra normalisation pass.
type keywordsPayload struct {
	High1 []string `json:"high_level_keywords"`
	Low1  []string `json:"low_level_keywords"`
	High2 []string `json:"highLevelKeywords"`
	Low2  []string `json:"lowLevelKeywords"`
}

// parseKeywordJSON robustly extracts the two arrays from the LLM response.
// Handles three common drifts:
//
//  1. Plain JSON (the contract).
//  2. Markdown-fenced JSON (```json ... ```).
//  3. JSON embedded in surrounding prose (we slice from the first '{' to the
//     matching final '}').
//
// Returns ([]string, []string, error).  Empty-but-valid is allowed.
func parseKeywordJSON(raw string) ([]string, []string, error) {
	body := strings.TrimSpace(raw)
	body = stripMarkdownFences(body)
	body = sliceJSONObject(body)
	if body == "" {
		return nil, nil, fmt.Errorf("no JSON object found")
	}
	var pl keywordsPayload
	if err := json.Unmarshal([]byte(body), &pl); err != nil {
		return nil, nil, err
	}
	hl := pl.High1
	if len(hl) == 0 {
		hl = pl.High2
	}
	ll := pl.Low1
	if len(ll) == 0 {
		ll = pl.Low2
	}
	return cleanList(hl), cleanList(ll), nil
}

// stripMarkdownFences removes a leading ```json / ``` and a trailing ```
// pair if present.  Idempotent: returns the input unchanged when no fences
// are found.
func stripMarkdownFences(s string) string {
	s = strings.TrimSpace(s)
	if strings.HasPrefix(s, "```") {
		// drop the first fence line
		nl := strings.IndexByte(s, '\n')
		if nl >= 0 {
			s = s[nl+1:]
		}
	}
	s = strings.TrimSpace(s)
	if strings.HasSuffix(s, "```") {
		s = strings.TrimSuffix(s, "```")
	}
	return strings.TrimSpace(s)
}

// sliceJSONObject extracts the first balanced JSON object substring from s.
// Returns "" if no balanced object is found.  Naive depth counter — does not
// understand strings, but the LLM output here is small and the trade-off
// favours simplicity over correctness on adversarial inputs.
func sliceJSONObject(s string) string {
	start := strings.IndexByte(s, '{')
	if start < 0 {
		return ""
	}
	depth := 0
	for i := start; i < len(s); i++ {
		switch s[i] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return s[start : i+1]
			}
		}
	}
	return ""
}

// cleanList trims whitespace from each element and drops empties.  Order
// preserved.  Used to give callers a tidy slice without nil-vs-empty pain.
func cleanList(xs []string) []string {
	out := make([]string, 0, len(xs))
	for _, x := range xs {
		x = strings.TrimSpace(x)
		if x != "" {
			out = append(out, x)
		}
	}
	return out
}
