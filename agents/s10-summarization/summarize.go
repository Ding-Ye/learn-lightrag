package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// summarize.go is the s10 core: turn N raw description fragments into ONE
// merged description, mirroring upstream's _handle_entity_relation_summary
// + _summarize_descriptions (lightrag/operate.py:167-385).
//
// Sessions are isolated — no cross-session imports.  All types below are
// re-declared with the same shape as the catalog in .learn/plan.md so a
// learner moving from s02 / s09 to s10 sees the same Provider / Entity /
// Relationship shapes show up.

// --- Provider abstraction (re-declared from s02) -----------------------------

// Provider is the LLM chat-completion abstraction (single method, just like
// every previous session).  s10 only needs Complete; the streaming knob in
// CompleteRequest is reserved for s11.
type Provider interface {
	Complete(ctx context.Context, req CompleteRequest) (CompleteResponse, error)
}

// CompleteRequest mirrors the catalog shape.
type CompleteRequest struct {
	Model       string
	System      string
	Messages    []Message
	MaxTokens   int
	Temperature float64
	Stream      bool
}

// Message is one turn in a chat conversation.
type Message struct {
	Role    string
	Content string
}

// CompleteResponse carries assistant text and token accounting.  s10 reads
// none of these fields — it just wants the text back.
type CompleteResponse struct {
	Text         string
	InputTokens  int
	OutputTokens int
}

// --- Domain types (re-declared from s09) -------------------------------------

// Entity is a knowledge-graph node with a single merged description.  Same
// shape as s09's Entity; re-declared because sessions are isolated.
type Entity struct {
	Name        string
	Type        string
	Description string
	SourceIDs   []string
}

// Relationship is a knowledge-graph edge.  Same shape as s09's Relationship.
type Relationship struct {
	SrcID, TgtID string
	Keywords     string
	Description  string
	Weight       float32
	SourceIDs    []string
}

// --- Defaults & tunables -----------------------------------------------------

// Defaults match upstream semantics:
//
//   - DefaultBudgetTokens (500)        upstream's summary_max_tokens.  Final
//                                       summary should fit under this.
//   - DefaultContextSize (2000)        upstream's summary_context_size.  Per-
//                                       chunk window when we map-split.
//   - DefaultCountThreshold (6)        upstream's force_llm_summary_on_merge.
//                                       Below this AND under budget = concat.
//   - DefaultMaxRecursionDepth (3)     our addition; upstream loops forever
//                                       under "while True" but trusts the
//                                       reduce phase to make progress.  We
//                                       cap recursion to 3 to avoid surprise
//                                       runaway costs in tests.
const (
	DefaultBudgetTokens       = 500
	DefaultContextSize        = 2000
	DefaultCountThreshold     = 6
	DefaultMaxRecursionDepth  = 3
	DefaultDescriptionType    = "entity"
	DefaultDescriptionName    = "(unnamed)"
	DefaultSummaryLanguage    = "English"
	descriptionJoinSeparator  = "\n\n"
)

// --- tokenCount stub ---------------------------------------------------------

// tokenCount is a SHOULD-BE-tiktoken stub.  s04 has the real cl100k_base
// tokenizer; s10 keeps a whitespace-word-count approximation so this
// session has zero external deps and can be reasoned about in isolation.
//
// The under-count is conservative: real tokens > word count for English
// (~1.3x for prose), so a budget that fits in word-count fits in real
// tokens too.  Document this on the README + every public function that
// reads tokenCount.
func tokenCount(s string) int {
	return len(strings.Fields(s))
}

// totalTokens sums tokenCount across a slice.
func totalTokens(descriptions []string) int {
	t := 0
	for _, d := range descriptions {
		t += tokenCount(d)
	}
	return t
}

// --- Public API: SummarizeDescriptions ---------------------------------------

// SummarizeDescriptions consolidates many raw description fragments into one
// merged description.  Mirrors upstream _handle_entity_relation_summary
// (lightrag/operate.py:167-303).  The decision tree:
//
//	if len(descriptions) == 0                                         → "", false
//	if len(descriptions) < countThreshold AND total < budgetTokens    → concat, false
//	otherwise                                                         → map-reduce LLM
//
// Map-reduce phase (when triggered):
//
//	1. Split descriptions into windows of contextSize tokens (>= 2 per chunk).
//	2. For each multi-description chunk, call Provider.Complete with the
//	   summaryPrompt; chunks of size 1 are passed through unchanged.
//	3. If the combined partial summaries still exceed budgetTokens, recurse
//	   on the partial summaries (up to DefaultMaxRecursionDepth).
//	4. Stop when result fits or recursion is exhausted (return what we have).
//
// budgetTokens defaults to DefaultBudgetTokens when <= 0.
// contextSize defaults to DefaultContextSize when <= 0.
// countThreshold defaults to DefaultCountThreshold when <= 0.
//
// Returns the merged summary text, a flag for "did we actually invoke the
// LLM?", and any error from the Provider.
func SummarizeDescriptions(
	ctx context.Context,
	p Provider,
	descriptions []string,
	budgetTokens int,
	contextSize int,
	countThreshold int,
) (string, bool, error) {
	if budgetTokens <= 0 {
		budgetTokens = DefaultBudgetTokens
	}
	if contextSize <= 0 {
		contextSize = DefaultContextSize
	}
	if countThreshold <= 0 {
		countThreshold = DefaultCountThreshold
	}
	return summarizeWithName(ctx, p, DefaultDescriptionType, DefaultDescriptionName,
		descriptions, budgetTokens, contextSize, countThreshold, 0)
}

// SummarizeDescriptionsForName is like SummarizeDescriptions but lets the
// caller pass through descriptionType ("entity" / "relation") and the
// canonical name — these flow into the LLM prompt for better grounding.
// Used by MergeEntities / MergeRelationships.
func SummarizeDescriptionsForName(
	ctx context.Context,
	p Provider,
	descriptionType string,
	descriptionName string,
	descriptions []string,
	budgetTokens int,
	contextSize int,
	countThreshold int,
) (string, bool, error) {
	if descriptionType == "" {
		descriptionType = DefaultDescriptionType
	}
	if descriptionName == "" {
		descriptionName = DefaultDescriptionName
	}
	if budgetTokens <= 0 {
		budgetTokens = DefaultBudgetTokens
	}
	if contextSize <= 0 {
		contextSize = DefaultContextSize
	}
	if countThreshold <= 0 {
		countThreshold = DefaultCountThreshold
	}
	return summarizeWithName(ctx, p, descriptionType, descriptionName,
		descriptions, budgetTokens, contextSize, countThreshold, 0)
}

// summarizeWithName is the recursive worker.  depth tracks current
// recursion level (capped at DefaultMaxRecursionDepth).
func summarizeWithName(
	ctx context.Context,
	p Provider,
	descriptionType string,
	descriptionName string,
	descriptions []string,
	budgetTokens int,
	contextSize int,
	countThreshold int,
	depth int,
) (string, bool, error) {
	// Step 0: cancellation check.
	if err := ctx.Err(); err != nil {
		return "", false, fmt.Errorf("summarize: %w", err)
	}

	// Step 1: edge cases.
	if len(descriptions) == 0 {
		return "", false, nil
	}
	if len(descriptions) == 1 {
		return descriptions[0], false, nil
	}

	// Step 2: heuristic — under threshold AND under budget → naive concat.
	// Mirrors upstream operate.py:222-226.
	total := totalTokens(descriptions)
	if len(descriptions) < countThreshold && total < budgetTokens {
		return strings.Join(descriptions, descriptionJoinSeparator), false, nil
	}

	// Step 3: map phase — chunk into contextSize-token windows.  Each window
	// gets >= 2 descriptions when possible (matches upstream's "minimum 2
	// per chunk" guarantee at operate.py:255-263 to ensure progress).
	chunks := chunkByTokens(descriptions, contextSize)

	// Step 4: reduce phase — summarize each chunk via Provider.  Single-
	// description chunks pass through unchanged (an upstream optimization
	// at operate.py:283-285).
	llmUsed := false
	partials := make([]string, 0, len(chunks))
	for _, chunk := range chunks {
		if err := ctx.Err(); err != nil {
			return "", llmUsed, fmt.Errorf("summarize reduce: %w", err)
		}
		if len(chunk) == 1 {
			partials = append(partials, chunk[0])
			continue
		}
		summary, err := callLLMSummary(ctx, p, descriptionType, descriptionName, chunk, budgetTokens)
		if err != nil {
			return "", llmUsed, fmt.Errorf("summarize chunk: %w", err)
		}
		partials = append(partials, summary)
		llmUsed = true
	}

	// Step 5: convergence check.  If the joined partials now fit the budget
	// and there are <= countThreshold of them, return the join.  Otherwise,
	// recurse to compress further.
	joined := strings.Join(partials, descriptionJoinSeparator)
	joinedTokens := tokenCount(joined)

	// If we collapsed to 1 partial and a single LLM call was made, return it.
	if len(partials) == 1 {
		return partials[0], llmUsed, nil
	}

	// Naive concat is acceptable now: under budget AND few enough.
	if joinedTokens < budgetTokens && len(partials) < countThreshold {
		return joined, llmUsed, nil
	}

	// Recursion guard: if we've recursed too deep, do one final LLM-summary
	// over what we have and return.  Better a coarse summary than infinite
	// recursion.
	if depth+1 >= DefaultMaxRecursionDepth {
		final, err := callLLMSummary(ctx, p, descriptionType, descriptionName, partials, budgetTokens)
		if err != nil {
			// Fall back to concat — caller still gets something usable.
			return joined, llmUsed, nil
		}
		return final, true, nil
	}

	// Step 6: recurse.  Pass partials back as new "descriptions".  llmUsed
	// stays true once any call has been made.
	finalSummary, recurseUsed, err := summarizeWithName(ctx, p, descriptionType,
		descriptionName, partials, budgetTokens, contextSize, countThreshold, depth+1)
	if err != nil {
		return "", llmUsed || recurseUsed, err
	}
	return finalSummary, llmUsed || recurseUsed, nil
}

// chunkByTokens splits descriptions into windows of <= contextSize tokens
// each, with a minimum of 2 descriptions per chunk where possible.  The
// "minimum 2" rule matches upstream operate.py:255-263 and guarantees the
// reduce phase makes progress (a 1-description chunk would just pass
// through, blocking convergence).
func chunkByTokens(descriptions []string, contextSize int) [][]string {
	if len(descriptions) == 0 {
		return nil
	}
	chunks := make([][]string, 0)
	current := make([]string, 0, 4)
	currentTokens := 0
	for _, desc := range descriptions {
		descTokens := tokenCount(desc)
		// If adding this would overflow AND current isn't empty, finalize.
		if currentTokens+descTokens > contextSize && len(current) > 0 {
			if len(current) == 1 {
				// Force one more so the chunk has 2 descriptions.
				current = append(current, desc)
				chunks = append(chunks, current)
				current = make([]string, 0, 4)
				currentTokens = 0
				continue
			}
			chunks = append(chunks, current)
			current = []string{desc}
			currentTokens = descTokens
			continue
		}
		current = append(current, desc)
		currentTokens += descTokens
	}
	if len(current) > 0 {
		chunks = append(chunks, current)
	}
	// Edge case: very large contextSize and few descriptions can leave us
	// with no chunks if descriptions is empty.  Guard above already handles
	// that.  Single-description input → 1 chunk of size 1; caller passes
	// through.
	return chunks
}

// callLLMSummary builds the summarization prompt, invokes the Provider, and
// returns the summary text.  Errors propagate up unchanged.
func callLLMSummary(
	ctx context.Context,
	p Provider,
	descriptionType string,
	descriptionName string,
	descriptions []string,
	budgetTokens int,
) (string, error) {
	if p == nil {
		return "", fmt.Errorf("summarize: nil Provider")
	}
	prompt := renderSummaryPrompt(descriptionType, descriptionName, descriptions, budgetTokens, DefaultSummaryLanguage)
	resp, err := p.Complete(ctx, CompleteRequest{
		System:      "",
		Messages:    []Message{{Role: "user", Content: prompt}},
		Temperature: 0.0,
	})
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(resp.Text), nil
}

// renderSummaryPrompt fills the summaryPrompt template with the given
// values.  We don't use text/template because there are only 5 placeholders
// and they're all string substitutions — strings.NewReplacer is faster and
// more obvious.
func renderSummaryPrompt(
	descriptionType string,
	descriptionName string,
	descriptions []string,
	summaryLength int,
	language string,
) string {
	descList := descriptionsToJSONL(descriptions)
	r := strings.NewReplacer(
		"{description_type}", descriptionType,
		"{description_name}", descriptionName,
		"{description_list}", descList,
		"{summary_length}", fmt.Sprintf("%d", summaryLength),
		"{language}", language,
	)
	return r.Replace(summaryPrompt)
}

// descriptionsToJSONL renders the descriptions as JSONL (one JSON object
// per line, each with a "Description" field) — matches upstream's input
// shape at lightrag/operate.py:325-327.  Encoding errors are unreachable
// for plain strings; we panic-fall-back to a fmt.Sprintf path that drops
// non-printable bytes rather than failing the call.
func descriptionsToJSONL(descriptions []string) string {
	var b strings.Builder
	for i, d := range descriptions {
		obj := map[string]string{"Description": d}
		raw, err := json.Marshal(obj)
		if err != nil {
			// Fallback — JSON marshal of a string-keyed map is essentially
			// infallible, but we keep the path defensive.
			fmt.Fprintf(&b, `{"Description":%q}`, d)
		} else {
			b.Write(raw)
		}
		if i < len(descriptions)-1 {
			b.WriteByte('\n')
		}
	}
	return b.String()
}
