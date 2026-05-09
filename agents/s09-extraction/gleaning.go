package main

import (
	"context"
	"fmt"
)

// gleaning.go runs the N-round continuation loop after the initial
// extraction.  Mirrors upstream's gleaning block at lightrag/operate.py:
// 2987-3050, with one didactic upgrade: upstream actually runs at most ONE
// extra round (entity_extract_max_gleaning is treated as a bool); we
// support N rounds with early-stop on zero-new-entity.
//
// Why the loop: LLMs hit a "stopping condition" too early on entity
// extraction — they emit 5-6 entities, decide they're done, output
// <|COMPLETE|>.  But the chunk may have 10.  Each gleaning round shows
// the LLM what it already extracted (via history_messages) and asks
// "what did you miss?".  Recall climbs from ~60% to ~85% with one round
// (paper-reported); the second round only adds ~5%.

// runGleaning runs maxRounds continuation prompts after the initial
// extraction.  prevEntities/prevRels are the cumulative results so far;
// returned entities/rels are the NEW additions plus the prevs (the loop
// merges as it goes so the caller gets the full set back).
//
// initialUserPrompt + initialAssistant are the (user, assistant) pair
// from the initial extraction round — these get packed into the message
// history every round so the LLM can see what was already extracted.
//
// system is the same system prompt used in the initial round (we keep it
// stable across rounds — only the user message changes per round, plus
// the assistant turns getting appended to history).
//
// Early-stop rule: if a round produces zero NEW entities, break.  This
// matches the empirical observation that gleaning yields drop steeply
// after the first round.
func runGleaning(
	ctx context.Context,
	p Provider,
	model string,
	system string,
	initialUserPrompt string,
	initialAssistant string,
	prevEntities []Entity,
	prevRels []Relationship,
	maxRounds int,
) ([]Entity, []Relationship, error) {
	if maxRounds <= 0 {
		return prevEntities, prevRels, nil
	}

	// Build the initial conversation history: user₀ + assistant₀.
	history := []Message{
		{Role: "user", Content: initialUserPrompt},
		{Role: "assistant", Content: initialAssistant},
	}

	// Index prev entities/relationships by their dedup key for fast
	// "is this new?" checks each round.
	seenEntity := make(map[string]struct{}, len(prevEntities))
	for _, e := range prevEntities {
		seenEntity[e.Name] = struct{}{}
	}
	seenRel := make(map[string]struct{}, len(prevRels))
	for _, r := range prevRels {
		seenRel[relKey(r)] = struct{}{}
	}

	entities := append([]Entity(nil), prevEntities...)
	rels := append([]Relationship(nil), prevRels...)

	for round := 0; round < maxRounds; round++ {
		// Respect ctx cancellation between rounds.
		if err := ctx.Err(); err != nil {
			return entities, rels, fmt.Errorf("gleaning round %d: %w", round+1, err)
		}

		// Build messages for this round: history + the continuation user prompt.
		msgs := make([]Message, 0, len(history)+1)
		msgs = append(msgs, history...)
		msgs = append(msgs, Message{Role: "user", Content: entityContinueExtractionUserPrompt})

		resp, err := p.Complete(ctx, CompleteRequest{
			Model:       model,
			System:      system,
			Messages:    msgs,
			Temperature: 0.0,
		})
		if err != nil {
			return entities, rels, fmt.Errorf("gleaning round %d: %w", round+1, err)
		}

		// Parse the round's output and fold in any new entities/rels.
		newEnts, newRels, perr := parseExtractionOutput(resp.Text)
		if perr != nil {
			return entities, rels, fmt.Errorf("gleaning round %d parse: %w", round+1, perr)
		}

		addedAny := false
		for _, e := range newEnts {
			if _, ok := seenEntity[e.Name]; ok {
				continue
			}
			seenEntity[e.Name] = struct{}{}
			entities = append(entities, e)
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

		// Append this round's user/assistant pair to the history so the
		// next round (if any) sees the full prior conversation.
		history = append(history,
			Message{Role: "user", Content: entityContinueExtractionUserPrompt},
			Message{Role: "assistant", Content: resp.Text},
		)

		// Early stop: no new entities this round.  We don't check
		// relationships because relations can lag entities by a round
		// (the LLM sometimes adds the entity in round 1 and the
		// relationship in round 2).
		if !addedAny {
			break
		}
	}

	return entities, rels, nil
}

// relKey is the canonical dedup key for a Relationship: undirected, so
// (A,B) and (B,A) map to the same key.  Matches the edgeKey trick from
// s08's adjacency_graph.go.
func relKey(r Relationship) string {
	a, b := r.SrcID, r.TgtID
	if a > b {
		a, b = b, a
	}
	return a + "\x00" + b
}
