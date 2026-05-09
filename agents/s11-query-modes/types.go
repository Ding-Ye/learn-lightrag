package main

import "context"

// types.go re-declares every shared type s11 needs.  Sessions are isolated:
// no learn-lightrag/sNN imports anywhere in this folder.  The shapes below
// are byte-for-byte identical to the catalog in .learn/plan.md and to what
// s02/s05/s06/s07/s08/s09/s10 each re-typed locally.  A learner moving from
// s10 to s11 sees the same Provider / Entity / Relationship / VectorStore /
// KVStore / GraphStore re-appear; s11 just picks them all up at once.

// --- LLM provider abstraction (re-declared from s02) -------------------------

// Provider is the chat-completion abstraction used by s11 for keyword
// extraction (one LLM call per non-naive query) and for the final synthesis
// call.  Single method; the streaming knob is reserved (s11 does not stream).
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

// CompleteResponse carries assistant text and token accounting.
type CompleteResponse struct {
	Text         string
	InputTokens  int
	OutputTokens int
}

// --- Embedding provider (re-declared from s06) -------------------------------

// EmbeddingProvider is the vectorizer abstraction.  s11 calls Embed once for
// the query (naive) plus once per keyword (local/global).  Dim() is read by
// the vector stores at construction time so each store knows its index width.
type EmbeddingProvider interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
	Dim() int
}

// --- Knowledge-graph types (re-declared from s09 / s10) ----------------------

// Entity is a knowledge-graph node — same shape as s09's Entity.
type Entity struct {
	Name        string
	Type        string
	Description string
	SourceIDs   []string
}

// Relationship is a knowledge-graph edge — same shape as s09's Relationship.
type Relationship struct {
	SrcID, TgtID string
	Keywords     string
	Description  string
	Weight       float32
	SourceIDs    []string
}

// Subgraph is what GraphStore.GetSubgraph returns — nodes + edges in one
// connected expansion of the seed.
type Subgraph struct {
	Nodes []Entity
	Edges []Relationship
}

// --- Storage interfaces (re-declared from s05 / s07 / s08) -------------------

// GraphStore is the adjacency-graph abstraction (s08).  s11 calls only
// GetSubgraph at query time; the upsert/persist path is exercised by
// ingestion in earlier sessions.
type GraphStore interface {
	UpsertNode(ctx context.Context, e Entity) error
	UpsertEdge(ctx context.Context, r Relationship) error
	GetNode(ctx context.Context, id string) (Entity, bool, error)
	GetSubgraph(ctx context.Context, seed string, maxDepth, maxNodes int) (Subgraph, error)
	Persist(ctx context.Context) error
}

// VectorRecord is the unit of storage in a vector index — id + vector +
// arbitrary metadata.  s11 stuffs entity_id / relation_key / chunk_id into
// metadata so a hit can be reverse-mapped back to a domain object.
type VectorRecord struct {
	ID       string
	Vector   []float32
	Metadata map[string]any
}

// VectorHit is one search result — id + similarity score + metadata copy.
type VectorHit struct {
	ID       string
	Score    float32
	Metadata map[string]any
}

// VectorStore is the cosine-similarity index abstraction (s07).  s11
// constructs three of them in the Pipeline: chunks / entities / relations.
type VectorStore interface {
	Upsert(ctx context.Context, records []VectorRecord) error
	Query(ctx context.Context, query []float32, topK int, threshold float32) ([]VectorHit, error)
	Delete(ctx context.Context, ids []string) error
	Persist(ctx context.Context) error
}

// KVStore is the chunk-content (and other JSON blob) store abstraction
// (s05).  s11 reads it via GetByIDs to fetch chunk content for SourceIDs
// surfaced through the graph.
type KVStore interface {
	Get(ctx context.Context, id string) (map[string]any, bool, error)
	GetByIDs(ctx context.Context, ids []string) (map[string]map[string]any, error)
	Upsert(ctx context.Context, items map[string]map[string]any) error
	FilterMissing(ctx context.Context, ids []string) ([]string, error)
	Delete(ctx context.Context, ids []string) error
	Persist(ctx context.Context) error
}

// --- Query API (this is the s11 contract) ------------------------------------

// QueryMode is one of the four retrieval strategies.  Naive is the s01
// baseline (vector-only); local / global / hybrid each blend graph context.
type QueryMode string

const (
	ModeNaive  QueryMode = "naive"
	ModeLocal  QueryMode = "local"
	ModeGlobal QueryMode = "global"
	ModeHybrid QueryMode = "hybrid"
)

// QueryParam tunes retrieval and budgeting per call.  TopK is the number
// of vector hits per index; ChunkTopK caps how many chunks survive into
// the final context window.  The three Max*Tokens fields are budgets for
// the entity / relation / total-prompt sections — see context_builder.go
// for how they compose.
type QueryParam struct {
	Mode              QueryMode
	TopK              int
	ChunkTopK         int
	MaxEntityTokens   int
	MaxRelationTokens int
	MaxTotalTokens    int
}

// QueryResult is what Query returns: synthesized answer text, the chunk
// IDs cited, and the mode that produced it (for downstream UX / logging).
type QueryResult struct {
	Content    string
	References []string
	Mode       QueryMode
}
