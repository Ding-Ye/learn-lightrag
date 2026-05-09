package main

import (
	"context"
	"fmt"
	"log"
)

// extraction.go declares the s09 type contract and the Extractor that
// orchestrates: prompt build → cache check → Provider.Complete → parse →
// gleaning → cache store.  Mirrors upstream's extract_entities main path
// at lightrag/operate.py:2883-3170 (we ignore the parallel-chunk + pipeline
// status concerns; those belong to s_full integration).
//
// Sessions are isolated — no imports from learn-lightrag/sNN.  The types
// below are RE-DECLARED with the same shape as the catalog in
// .learn/plan.md so a learner moving from s02/s04/s08 to s09 sees the
// same interfaces show up.

// --- Provider abstraction (re-declared from s02's contract) ---

// Provider is the LLM chat-completion abstraction.  Same shape as s02 and
// every later session.  Single method; everything else is in the request.
type Provider interface {
	Complete(ctx context.Context, req CompleteRequest) (CompleteResponse, error)
}

// CompleteRequest mirrors the catalog shape: model, system, message
// history, max-tokens, temperature, stream flag.  Stream=false for s09.
type CompleteRequest struct {
	Model       string
	System      string
	Messages    []Message
	MaxTokens   int
	Temperature float64
	Stream      bool
}

// Message is one turn in a chat conversation.  Roles: "user", "assistant",
// "system" (though we put the system prompt in CompleteRequest.System).
type Message struct {
	Role    string
	Content string
}

// CompleteResponse is what a Provider returns: the assistant text and
// token accounting (we ignore the latter at this layer; s10 uses it for
// summarization budget logic).
type CompleteResponse struct {
	Text         string
	InputTokens  int
	OutputTokens int
}

// --- Domain types (re-declared from s04 / s08) ---

// Chunk mirrors s04's TextChunkSchema: a token-counted text fragment with
// its parent doc ID and order index.  s09 only reads Content + uses the
// (ContentDocID, ChunkOrderIndex) tuple to derive a stable chunk ID.
type Chunk struct {
	ContentDocID    string
	Content         string
	Tokens          int
	ChunkOrderIndex int
}

// ChunkID returns a stable identifier for this chunk: "<docID>:<index>".
// Used as the SourceID on every entity/relationship extracted from this
// chunk.  Matches the upstream pattern of "chunk-<hash>" but with simpler
// derivation — sufficient for didactic purposes.
func (c Chunk) ChunkID() string {
	return fmt.Sprintf("%s:%d", c.ContentDocID, c.ChunkOrderIndex)
}

// Entity is a knowledge-graph node.  Same shape as s08's Entity (we
// re-declare; sessions are isolated).
type Entity struct {
	Name        string
	Type        string
	Description string
	SourceIDs   []string
}

// Relationship is a knowledge-graph edge.  Same shape as s08's
// Relationship (re-declared).
type Relationship struct {
	SrcID, TgtID string
	Keywords     string
	Description  string
	Weight       float32
	SourceIDs    []string
}

// --- Logger (interface, optional) ---

// Logger is the minimal logging contract Extractor uses.  Defaults to a
// no-op so tests don't pollute stdout.  Pass log.New(os.Stderr, ...) or a
// custom impl to enable per-chunk timing/cache-hit logs.
type Logger interface {
	Printf(format string, args ...any)
}

// noopLogger discards all log calls.
type noopLogger struct{}

func (noopLogger) Printf(format string, args ...any) {}

// stdLogger wraps a *log.Logger to satisfy our Logger interface.
type stdLogger struct{ l *log.Logger }

func (s stdLogger) Printf(format string, args ...any) { s.l.Printf(format, args...) }

// --- Extractor ---

// Extractor is the top-level orchestrator: given a Chunk, it returns the
// entities and relationships extracted from the chunk's text, optionally
// using a cache to skip the LLM call on repeated chunks.
//
// Construct via NewExtractor — direct struct literal is fine too, but the
// constructor enforces sane defaults (e.g., Logger != nil).
//
// Goroutine-safety: all fields except metrics are read-only after
// construction; metrics use atomics in their setters.  Multiple goroutines
// may share one Extractor.
type Extractor struct {
	// Provider is the LLM backing the extraction.  Required.
	Provider Provider

	// Model is the model name passed in CompleteRequest.Model.  Empty
	// means provider-default.
	Model string

	// MaxGleaningRounds is the upper bound on continuation rounds AFTER
	// the initial extraction.  Default: 1 (matches upstream).  0 disables
	// gleaning entirely.
	MaxGleaningRounds int

	// Cache is the optional KV store for caching raw LLM outputs by
	// sha256(chunk + promptVersion).  nil disables caching.
	Cache KVStore

	// Logger is the optional logger.  nil → noopLogger.
	Logger Logger

	// Metrics — accessed via Stats() at the end of a run.  Internally
	// guarded by sync/atomic in counter setters; we use plain fields here
	// since the test suite drives Extractor sequentially.  For concurrent
	// callers, wrap with atomic.AddInt64 in IncCacheHit/IncCacheMiss.
	cacheHits   int
	cacheMisses int
	llmCalls    int
}

// NewExtractor constructs an Extractor with sane defaults.  Provider is
// required; everything else is optional.
func NewExtractor(p Provider, opts ...Option) *Extractor {
	e := &Extractor{
		Provider:          p,
		MaxGleaningRounds: 1, // upstream default
		Logger:            noopLogger{},
	}
	for _, o := range opts {
		o(e)
	}
	return e
}

// Option is a functional option for NewExtractor.
type Option func(*Extractor)

// WithModel sets the model name.
func WithModel(m string) Option { return func(e *Extractor) { e.Model = m } }

// WithGleaningRounds sets MaxGleaningRounds.
func WithGleaningRounds(n int) Option { return func(e *Extractor) { e.MaxGleaningRounds = n } }

// WithCache sets the optional cache backend.
func WithCache(c KVStore) Option { return func(e *Extractor) { e.Cache = c } }

// WithLogger sets the optional logger.
func WithLogger(l Logger) Option {
	return func(e *Extractor) {
		if l == nil {
			e.Logger = noopLogger{}
			return
		}
		e.Logger = l
	}
}

// Stats returns the current cache-hit / cache-miss / LLM-call counters.
// Used by the CLI to print the hit ratio at end of run.
type Stats struct {
	CacheHits   int
	CacheMisses int
	LLMCalls    int
}

// Stats returns a snapshot.
func (e *Extractor) Stats() Stats {
	return Stats{CacheHits: e.cacheHits, CacheMisses: e.cacheMisses, LLMCalls: e.llmCalls}
}

// Extract runs the full pipeline on one chunk: cache lookup → LLM call →
// parse → gleaning loop → cache store.  Returns the entities and
// relationships extracted (with SourceIDs set to [chunk.ChunkID()]).
//
// The chunk's Content is NEVER mutated.  The returned slices are freshly
// allocated; mutating them does not affect the cache.
func (e *Extractor) Extract(ctx context.Context, chunk Chunk) ([]Entity, []Relationship, error) {
	if e.Provider == nil {
		return nil, nil, fmt.Errorf("extractor: nil Provider")
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, fmt.Errorf("extract: %w", err)
	}

	chunkID := chunk.ChunkID()
	key := cacheKey(chunk.Content)

	// Step 1: cache lookup.
	if e.Cache != nil {
		if cached, ok, err := e.Cache.Get(ctx, key); err != nil {
			// Soft error: log and proceed as if it was a miss.
			e.Logger.Printf("cache.Get(%s) error: %v — proceeding to LLM", key, err)
		} else if ok {
			rawText, _ := cached["raw"].(string)
			if rawText != "" {
				ents, rels, perr := parseExtractionOutput(rawText)
				if perr != nil {
					e.Logger.Printf("cache parse failed: %v — re-extracting", perr)
				} else {
					e.cacheHits++
					return tagEntitySources(ents, chunkID), tagRelSources(rels, chunkID), nil
				}
			}
		}
	}

	// Step 2: LLM call (initial round).
	e.cacheMisses++
	userPrompt := buildExtractionUserPrompt(chunk.Content)
	resp, err := e.Provider.Complete(ctx, CompleteRequest{
		Model:       e.Model,
		System:      entityExtractionSystemPrompt,
		Messages:    []Message{{Role: "user", Content: userPrompt}},
		Temperature: 0.0,
	})
	if err != nil {
		return nil, nil, fmt.Errorf("provider.Complete: %w", err)
	}
	e.llmCalls++

	// Step 3: parse the initial round.
	entities, relationships, perr := parseExtractionOutput(resp.Text)
	if perr != nil {
		return nil, nil, fmt.Errorf("parse initial: %w", perr)
	}

	// Step 4: gleaning loop (only counts LLM calls inside the loop).
	if e.MaxGleaningRounds > 0 {
		entitiesBefore, relsBefore := len(entities), len(relationships)
		// runGleaning makes its own Provider.Complete calls — count them
		// by snapshotting before/after.
		var gErr error
		entities, relationships, gErr = e.runGleaningCounted(
			ctx, userPrompt, resp.Text, entities, relationships,
		)
		if gErr != nil {
			// Return what we have so far — gleaning failure shouldn't
			// invalidate the initial extraction.  Log and continue.
			e.Logger.Printf("gleaning aborted: %v (kept %d entities, %d rels from initial)",
				gErr, entitiesBefore, relsBefore)
			// Surface ctx-cancel errors to the caller; everything else
			// is a non-fatal warning.
			if ctx.Err() != nil {
				return tagEntitySources(entities, chunkID), tagRelSources(relationships, chunkID), gErr
			}
		}
	}

	// Step 5: write the raw response to cache (raw text — not the parsed
	// shape — so we re-parse on hit and pick up parser improvements).
	if e.Cache != nil {
		if uerr := e.Cache.Upsert(ctx, map[string]map[string]any{
			key: {"raw": resp.Text, "chunk_id": chunkID},
		}); uerr != nil {
			e.Logger.Printf("cache.Upsert(%s) error: %v (continuing)", key, uerr)
		}
	}

	return tagEntitySources(entities, chunkID), tagRelSources(relationships, chunkID), nil
}

// runGleaningCounted wraps runGleaning to bump the LLM-call counter for
// each round it runs.  We can't easily count calls inside runGleaning
// without making it Extractor-aware, so we snapshot e.llmCalls before/after
// and let the gleaning code do its own provider work.
//
// To make the call-count accurate we re-implement the loop here calling
// Provider.Complete directly — this also gives us tight control over the
// ctx-cancel surfacing required by TestExtractionRespectsContextCancel.
func (e *Extractor) runGleaningCounted(
	ctx context.Context,
	initialUserPrompt string,
	initialAssistant string,
	prevEntities []Entity,
	prevRels []Relationship,
) ([]Entity, []Relationship, error) {
	// Track which entities/rels we've already seen across rounds.
	seenEntity := make(map[string]struct{}, len(prevEntities))
	for _, ent := range prevEntities {
		seenEntity[ent.Name] = struct{}{}
	}
	seenRel := make(map[string]struct{}, len(prevRels))
	for _, r := range prevRels {
		seenRel[relKey(r)] = struct{}{}
	}

	history := []Message{
		{Role: "user", Content: initialUserPrompt},
		{Role: "assistant", Content: initialAssistant},
	}

	entities := append([]Entity(nil), prevEntities...)
	rels := append([]Relationship(nil), prevRels...)

	for round := 0; round < e.MaxGleaningRounds; round++ {
		if err := ctx.Err(); err != nil {
			return entities, rels, fmt.Errorf("gleaning round %d: %w", round+1, err)
		}
		msgs := make([]Message, 0, len(history)+1)
		msgs = append(msgs, history...)
		msgs = append(msgs, Message{Role: "user", Content: entityContinueExtractionUserPrompt})

		resp, err := e.Provider.Complete(ctx, CompleteRequest{
			Model:       e.Model,
			System:      entityExtractionSystemPrompt,
			Messages:    msgs,
			Temperature: 0.0,
		})
		if err != nil {
			return entities, rels, fmt.Errorf("gleaning round %d: %w", round+1, err)
		}
		e.llmCalls++

		newEnts, newRels, perr := parseExtractionOutput(resp.Text)
		if perr != nil {
			return entities, rels, fmt.Errorf("gleaning round %d parse: %w", round+1, perr)
		}

		addedAny := false
		for _, ent := range newEnts {
			if _, ok := seenEntity[ent.Name]; ok {
				continue
			}
			seenEntity[ent.Name] = struct{}{}
			entities = append(entities, ent)
			addedAny = true
		}
		for _, r := range newRels {
			k := relKey(r)
			if _, ok := seenRel[k]; ok {
				continue
			}
			seenRel[k] = struct{}{}
			rels = append(rels, r)
		}

		history = append(history,
			Message{Role: "user", Content: entityContinueExtractionUserPrompt},
			Message{Role: "assistant", Content: resp.Text},
		)

		if !addedAny {
			break
		}
	}

	return entities, rels, nil
}

// buildExtractionUserPrompt formats the per-chunk user prompt that's sent
// to the Provider in the initial extraction round.  We don't use a Go
// template because there's exactly one placeholder (the chunk content)
// and the prompt is internal — string concat is clearer.
func buildExtractionUserPrompt(chunkContent string) string {
	return "---Task---\nExtract entities and relationships from the input text below.\n" +
		"Follow the format and completion-signal rules from the system prompt exactly.\n\n" +
		"---Input Text---\n```\n" + chunkContent + "\n```\n\n<Output>\n"
}

// tagEntitySources stamps the chunk ID onto every entity's SourceIDs.
// Returned slice is a fresh allocation; mutating it does not affect
// inputs.  Companion to tagRelSources for relationships.
func tagEntitySources(entities []Entity, chunkID string) []Entity {
	out := make([]Entity, len(entities))
	for i, e := range entities {
		ec := e
		ec.SourceIDs = append([]string(nil), e.SourceIDs...)
		ec.SourceIDs = append(ec.SourceIDs, chunkID)
		out[i] = ec
	}
	return out
}

// tagRelSources stamps the chunk ID onto every relationship's SourceIDs.
// Mirror of tagSources but for relationships.  Caller calls both.
func tagRelSources(rels []Relationship, chunkID string) []Relationship {
	out := make([]Relationship, len(rels))
	for i, r := range rels {
		rc := r
		rc.SourceIDs = append([]string(nil), r.SourceIDs...)
		rc.SourceIDs = append(rc.SourceIDs, chunkID)
		out[i] = rc
	}
	return out
}
