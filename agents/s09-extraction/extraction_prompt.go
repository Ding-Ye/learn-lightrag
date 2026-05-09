package main

// extraction_prompt.go holds the system-prompt + continuation-prompt
// constants for s09.  These are FAITHFUL SUMMARIES of the upstream prompts
// at lightrag/prompt.py:11-100, NOT verbatim copies — the upstream prompts
// are ~70 lines each with formatting placeholders ({entity_types}, etc.) we
// don't need at this layer.  The summaries preserve every load-bearing
// instruction (delimiter contract, role definition, completion signal,
// undirected-relationship rule) and use the SAME delimiter strings so the
// parser in parser.go matches upstream output.
//
// Delimiter constants are exported (lowercase here, package-internal) so
// parser.go and gleaning.go all see the same source of truth.  Bumping
// promptVersion invalidates every cache entry — change it whenever you
// edit either prompt below.
//
// Upstream cite (verified 2026-05-09):
//   lightrag/prompt.py:8                DEFAULT_TUPLE_DELIMITER      "<|#|>"
//   lightrag/prompt.py:9                DEFAULT_COMPLETION_DELIMITER "<|COMPLETE|>"
//   lightrag/prompt.py:11-61            entity_extraction_system_prompt
//   lightrag/prompt.py:84-100           entity_continue_extraction_user_prompt

const (
	// tupleDelimiter splits fields within ONE record.  Identical to upstream's
	// PROMPTS["DEFAULT_TUPLE_DELIMITER"].  Unlikely to appear in natural text.
	tupleDelimiter = "<|#|>"

	// completionDelimiter marks the end of the LLM output block.  Identical
	// to upstream's PROMPTS["DEFAULT_COMPLETION_DELIMITER"].  Stripped by
	// the parser before line-split — its absence is logged but tolerated.
	completionDelimiter = "<|COMPLETE|>"

	// promptVersion gets concatenated to the chunk content before hashing
	// to form the cache key.  Bump this when either prompt below changes
	// to invalidate stale cache entries.
	promptVersion = "v1"
)

// entityExtractionSystemPrompt is a faithful summary of upstream's
// entity_extraction_system_prompt (lightrag/prompt.py:11-61).  We omit
// the {language} / {entity_types} / {examples} formatting holes — the
// learner just needs to see "this is a knowledge-graph specialist with a
// strict delimiter contract".  Delimiters are interpolated literally below
// so the output the LLM produces matches what parser.go expects.
const entityExtractionSystemPrompt = `---Role---
You are a Knowledge Graph Specialist responsible for extracting entities
and relationships from input text.

---Instructions---
1. Entity extraction:
   - Identify clearly defined and meaningful entities in the input text.
   - For each entity, extract: entity_name, entity_type
     (PERSON / ORG / LOCATION / EVENT / CONCEPT / OTHER), entity_description.
   - Output 4 fields delimited by ` + tupleDelimiter + `, on a single line.
     The first field MUST be the literal string "entity".
     Format: entity` + tupleDelimiter + `name` + tupleDelimiter + `type` + tupleDelimiter + `description

2. Relationship extraction:
   - Identify direct, clearly stated relationships between extracted entities.
   - Decompose N-ary relationships into binary pairs.
   - For each binary relationship, extract: source_entity, target_entity,
     relationship_keywords (comma-separated, do NOT use the field delimiter
     within keywords), relationship_description.
   - Output 5 fields delimited by ` + tupleDelimiter + `, on a single line.
     The first field MUST be the literal string "relation".
     Format: relation` + tupleDelimiter + `src` + tupleDelimiter + `tgt` + tupleDelimiter + `keywords` + tupleDelimiter + `description

3. Delimiter usage:
   - The delimiter ` + tupleDelimiter + ` is atomic; never put content into it.
   - Treat all relationships as undirected unless stated otherwise.
   - Do not output duplicate relationships.

4. Output ordering:
   - Output all entities first, then all relationships.
   - Within relationships, prioritize those most central to the input.

5. Style:
   - Third-person; explicit subjects (avoid "this article", "we", pronouns).
   - Proper nouns retain original language if no widely-accepted translation exists.

6. Completion:
   - Output the literal string ` + completionDelimiter + ` only after every
     entity and relationship has been emitted.

(Upstream cite: lightrag/prompt.py:11-61, summarized — the full upstream
prompt also includes role-specific examples per entity-type schema.)`

// entityContinueExtractionUserPrompt is a faithful summary of upstream's
// entity_continue_extraction_user_prompt (lightrag/prompt.py:84-100).
// Sent on each gleaning round AFTER the initial extraction.  The previous
// (user, assistant) pair is packed into Messages so the LLM sees what was
// already extracted — this prompt only asks for the delta.
const entityContinueExtractionUserPrompt = `---Task---
Based on the previous extraction, identify and extract any MISSED or
INCORRECTLY FORMATTED entities and relationships from the input text.

---Instructions---
1. Strict adherence: follow the same delimiter contract from the system prompt.
2. Focus on additions/corrections:
   - Do NOT re-output entities/relationships that were correctly extracted.
   - If something was missed, extract it now in the specified format.
   - If a record was truncated or had missing fields, re-output the corrected
     version.
3. Entity format: 4 fields delimited by ` + tupleDelimiter + `, first field "entity".
4. Relation format: 5 fields delimited by ` + tupleDelimiter + `, first field "relation".
5. Output ONLY the additional/corrected list — no preamble, no commentary.
6. Completion: emit ` + completionDelimiter + ` as the final line.

(Upstream cite: lightrag/prompt.py:84-100, summarized.)`
