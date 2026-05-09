package main

import (
	"strings"
)

// parser.go implements the delimiter-based extraction-output parser.
// Mirrors upstream's _process_extraction_result + _handle_single_entity_extraction
// + _handle_single_relationship_extraction at lightrag/operate.py:937-540.
//
// Output format the LLM is supposed to produce (one record per line):
//
//   entity<|#|>NAME<|#|>TYPE<|#|>DESCRIPTION
//   entity<|#|>NAME2<|#|>TYPE2<|#|>DESCRIPTION2
//   relation<|#|>SRC<|#|>TGT<|#|>KEYWORDS<|#|>DESCRIPTION
//   <|COMPLETE|>
//
// Robustness rule #1: malformed lines are SKIPPED, not errored on.  An LLM
// that drops a field on one entity should not poison the whole chunk's
// output.  Robustness rule #2: be lenient about the prefix — treat
// "entity", "(entity", `"entity"` all as the entity record kind.  Same for
// "relation" / "relationship".
//
// We deliberately do NOT implement upstream's "repair" path (rewriting
// lines that used <|#|> as a record separator instead of \n).  In practice
// modern LLMs (gpt-4o-mini, claude-3.5+, llama-3.1+) do not emit that
// failure mode often enough to justify the ~30 LOC of repair logic.
// Permissive parsing is enough — see TestExtractionMalformedOutputFallsBack.

// parseExtractionOutput parses a delimited LLM extraction response into
// entities and relationships.  Always returns nil error in the current
// impl — the error return is reserved for future strict-mode validation
// (e.g., reject when 0 records parse despite a non-empty input).
//
// SourceIDs is left nil: the caller (Extractor.Extract) is responsible
// for populating it based on the chunk being extracted from.
func parseExtractionOutput(s string) ([]Entity, []Relationship, error) {
	// Step 1: strip the completion marker (case-insensitive — upstream
	// also tolerates "<|complete|>").  We don't error if it's missing;
	// upstream just logs a warning and continues.
	cleaned := s
	cleaned = strings.ReplaceAll(cleaned, completionDelimiter, "")
	cleaned = strings.ReplaceAll(cleaned, strings.ToLower(completionDelimiter), "")

	var entities []Entity
	var relationships []Relationship

	// Step 2: split records by newline.  Some LLMs emit \r\n on Windows
	// or extra blank lines — Split handles both since we trim and skip
	// empty lines below.
	for _, raw := range strings.Split(cleaned, "\n") {
		line := strings.TrimSpace(raw)
		if line == "" {
			continue
		}

		// Some LLMs wrap records in parens: "(entity<|#|>...)" — strip
		// any surrounding parens / quotes before splitting.
		line = strings.Trim(line, "()\"' \t")
		if line == "" {
			continue
		}

		// Step 3: split fields by tuple delimiter.
		fields := strings.Split(line, tupleDelimiter)
		if len(fields) < 2 {
			// Not even (kind + 1 field) — definitely not a record.
			continue
		}

		// Lowercase the kind field and strip any leftover punctuation
		// from the prefix so "entity" / `"entity"` / `(entity` all match.
		kind := strings.ToLower(strings.TrimSpace(fields[0]))
		kind = strings.Trim(kind, "()\"' \t")

		switch {
		case strings.Contains(kind, "entity") && len(fields) >= 4:
			ent := buildEntity(fields)
			if ent.Name == "" {
				// Empty name after sanitization — skip silently
				// (upstream does the same at operate.py:407).
				continue
			}
			entities = append(entities, ent)

		case (strings.Contains(kind, "relation") || strings.Contains(kind, "relationship")) && len(fields) >= 5:
			rel := buildRelationship(fields)
			if rel.SrcID == "" || rel.TgtID == "" {
				continue
			}
			relationships = append(relationships, rel)

		default:
			// Wrong prefix or insufficient fields — drop and continue.
			continue
		}
	}

	return entities, relationships, nil
}

// buildEntity converts a 4+ field record into an Entity.  Trims each field
// and lowercases Type (upstream operate.py:441 also lowercases).  Strips
// surrounding quotes from Name to handle `"entity"<|#|>"Tokyo"<|#|>...`.
func buildEntity(fields []string) Entity {
	name := strings.TrimSpace(fields[1])
	name = strings.Trim(name, "\"' \t")

	typ := strings.TrimSpace(fields[2])
	typ = strings.Trim(typ, "\"' \t")
	typ = strings.ToLower(typ)

	desc := strings.TrimSpace(fields[3])
	desc = strings.Trim(desc, "\"' \t")

	return Entity{
		Name:        name,
		Type:        typ,
		Description: desc,
		// SourceIDs intentionally nil — caller fills in the chunk ID.
	}
}

// buildRelationship converts a 5+ field record into a Relationship.  Trims
// each field; sets Weight to 1.0 (upstream defaults to 1.0 unless the LLM
// specifies otherwise — we don't parse a weight field at this layer).
func buildRelationship(fields []string) Relationship {
	src := strings.TrimSpace(fields[1])
	src = strings.Trim(src, "\"' \t")

	tgt := strings.TrimSpace(fields[2])
	tgt = strings.Trim(tgt, "\"' \t")

	keywords := strings.TrimSpace(fields[3])
	keywords = strings.Trim(keywords, "\"' \t")

	desc := strings.TrimSpace(fields[4])
	desc = strings.Trim(desc, "\"' \t")

	return Relationship{
		SrcID:       src,
		TgtID:       tgt,
		Keywords:    keywords,
		Description: desc,
		Weight:      1.0,
		// SourceIDs intentionally nil — caller fills in the chunk ID.
	}
}
