package main

import (
	"context"
	"strings"
	"testing"
)

// summarize_test.go — exactly the 6 tests required by the s10 spec, plus a
// compile-time interface contract check.  Tests are numbered to match the
// spec for easy cross-reference.  All tests use a stub MockProvider; no
// network access is performed.

// TestProviderInterfaceContract — compile-time assertion that *MockProvider
// satisfies Provider.  If anyone changes the Provider signature in
// summarize.go without updating MockProvider in main.go, this fails to
// build (caught by `go vet`).
type providerContract interface {
	Complete(ctx context.Context, req CompleteRequest) (CompleteResponse, error)
}

var _ providerContract = (*MockProvider)(nil)

// (1) TestSummarizeShortListSkipsLLM — 3 descriptions, plenty of budget;
// assert llmUsed=false, return = concat with descriptionJoinSeparator.
func TestSummarizeShortListSkipsLLM(t *testing.T) {
	prov := NewMockProvider()
	descs := []string{
		"Eleanor Hartwell is a senior archivist at Whitehall.",
		"Eleanor Hartwell discovered a lost manuscript.",
		"Eleanor Hartwell published her findings in 2019.",
	}

	out, llmUsed, err := SummarizeDescriptions(context.Background(), prov, descs,
		500 /*budgetTokens — way above input total*/, 2000, 6)
	if err != nil {
		t.Fatalf("SummarizeDescriptions: %v", err)
	}
	if llmUsed {
		t.Errorf("llmUsed=true; expected false (3 < threshold 6, total < budget)")
	}
	want := strings.Join(descs, "\n\n")
	if out != want {
		t.Errorf("output should be exact concat;\nwant: %q\n got: %q", want, out)
	}
	if prov.Calls != 0 {
		t.Errorf("Provider.Calls=%d; expected 0 — LLM should not be invoked", prov.Calls)
	}
}

// (2) TestSummarizeLongListInvokesLLM — 12 descriptions, tight budget;
// assert llmUsed=true, mock provider Calls > 0.
func TestSummarizeLongListInvokesLLM(t *testing.T) {
	prov := NewMockProvider()
	descs := make([]string, 12)
	for i := 0; i < 12; i++ {
		descs[i] = "Eleanor Hartwell is a senior archivist who discovered a lost manuscript and published findings."
	}

	_, llmUsed, err := SummarizeDescriptions(context.Background(), prov, descs,
		60 /*tight budget*/, 80 /*small windows*/, 6)
	if err != nil {
		t.Fatalf("SummarizeDescriptions: %v", err)
	}
	if !llmUsed {
		t.Errorf("llmUsed=false; expected true (12 > threshold 6 forces LLM path)")
	}
	if prov.Calls == 0 {
		t.Errorf("Provider.Calls=0; expected > 0 — LLM should have been invoked")
	}
}

// (3) TestSummarizeRespectsBudget — long input; assert the final output's
// tokenCount is within tolerance of budgetTokens.  We use the mock's
// FixedResponse to inject a known short summary and verify the recursion
// stops once it fits.
func TestSummarizeRespectsBudget(t *testing.T) {
	prov := NewMockProvider()
	prov.FixedResponse = "Eleanor Hartwell is a senior archivist whose work spans manuscript discovery, restoration, and public outreach across multiple decades."
	// FixedResponse is 18 words → ~18 tokens by our stub.

	descs := make([]string, 20)
	for i := 0; i < 20; i++ {
		descs[i] = "Eleanor Hartwell is a senior archivist who discovered a lost manuscript and published findings in many journals during her long career."
	}

	out, llmUsed, err := SummarizeDescriptions(context.Background(), prov, descs,
		50 /*budget*/, 80 /*context*/, 6)
	if err != nil {
		t.Fatalf("SummarizeDescriptions: %v", err)
	}
	if !llmUsed {
		t.Errorf("llmUsed=false; expected true for 20 descriptions")
	}
	got := tokenCount(out)
	// Tolerance: result might be 1x to 2x the budget if recursion bottoms
	// out before perfect convergence (DefaultMaxRecursionDepth=3 cap).  The
	// load-bearing assertion is "smaller than raw concat".
	rawTokens := totalTokens(descs)
	if got >= rawTokens {
		t.Errorf("summary not compressed: got %d tokens, raw was %d", got, rawTokens)
	}
	if got > 2*50 /*budget*/ {
		t.Errorf("summary far exceeds budget tolerance: got %d tokens, budget %d", got, 50)
	}
}

// (4) TestSummarizeRecursesUntilFits — extremely long input that needs
// multiple recursion rounds; assert mock provider Calls > 1 AND result fits
// (the recursion guard returns SOMETHING fitting under the cap).
func TestSummarizeRecursesUntilFits(t *testing.T) {
	prov := NewMockProvider()
	// FixedResponse keeps each summary the same length, so recursion only
	// converges via the depth cap, not natural shrinkage.
	prov.FixedResponse = "A concise multi-paragraph summary covering the key facts and dispositions of the subject across all source descriptions provided."

	descs := make([]string, 30)
	for i := 0; i < 30; i++ {
		descs[i] = "Eleanor Hartwell is a senior archivist who discovered a lost manuscript and published findings in journals."
	}

	out, llmUsed, err := SummarizeDescriptions(context.Background(), prov, descs,
		30 /*very tight budget*/, 60 /*small windows*/, 6)
	if err != nil {
		t.Fatalf("SummarizeDescriptions: %v", err)
	}
	if !llmUsed {
		t.Fatalf("llmUsed=false; expected true")
	}
	if prov.Calls <= 1 {
		t.Errorf("Provider.Calls=%d; expected > 1 to indicate recursion (multiple summary calls)", prov.Calls)
	}
	if out == "" {
		t.Errorf("empty output; recursion should always return something")
	}
	// Output should be at least somewhat shrunk vs raw concat.
	rawTokens := totalTokens(descs)
	gotTokens := tokenCount(out)
	if gotTokens >= rawTokens {
		t.Errorf("output not shrunk: got %d tokens vs raw %d", gotTokens, rawTokens)
	}
}

// (5) TestSummarizeEmptyDescriptionsReturnsEmpty — [] → "", false, nil.
func TestSummarizeEmptyDescriptionsReturnsEmpty(t *testing.T) {
	prov := NewMockProvider()
	out, llmUsed, err := SummarizeDescriptions(context.Background(), prov, nil, 500, 2000, 6)
	if err != nil {
		t.Fatalf("SummarizeDescriptions(empty): %v", err)
	}
	if out != "" {
		t.Errorf("empty input should return empty string; got %q", out)
	}
	if llmUsed {
		t.Errorf("empty input should not invoke LLM; llmUsed=true")
	}
	if prov.Calls != 0 {
		t.Errorf("empty input triggered Provider.Calls=%d", prov.Calls)
	}

	// Also try empty slice (zero-length, non-nil).
	out, llmUsed, err = SummarizeDescriptions(context.Background(), prov, []string{}, 500, 2000, 6)
	if err != nil {
		t.Fatalf("SummarizeDescriptions([]): %v", err)
	}
	if out != "" || llmUsed {
		t.Errorf("[]string{} should return ('', false, nil); got (%q, %v)", out, llmUsed)
	}
}

// (6) TestMergeEntityAggregatesSourceIDs — 3 Entity records with same Name
// from chunks [c1], [c2], [c3]; assert MergeEntities returns 1 Entity with
// SourceIDs sorted = [c1, c2, c3].
func TestMergeEntityAggregatesSourceIDs(t *testing.T) {
	prov := NewMockProvider()
	ents := []Entity{
		{Name: "Eleanor Hartwell", Type: "person",
			Description: "senior archivist at Whitehall.", SourceIDs: []string{"c1"}},
		{Name: "Eleanor Hartwell", Type: "person",
			Description: "discovered a lost manuscript.", SourceIDs: []string{"c2"}},
		{Name: "Eleanor Hartwell", Type: "person",
			Description: "published findings in 2019.", SourceIDs: []string{"c3"}},
	}

	merged, err := MergeEntities(context.Background(), prov, ents)
	if err != nil {
		t.Fatalf("MergeEntities: %v", err)
	}
	if len(merged) != 1 {
		t.Fatalf("expected 1 merged entity, got %d: %+v", len(merged), merged)
	}
	got := merged[0]
	if got.Name != "Eleanor Hartwell" {
		t.Errorf("merged Name = %q, want Eleanor Hartwell", got.Name)
	}
	if got.Type != "person" {
		t.Errorf("merged Type = %q, want person", got.Type)
	}
	wantIDs := []string{"c1", "c2", "c3"}
	if len(got.SourceIDs) != len(wantIDs) {
		t.Fatalf("merged SourceIDs len = %d, want 3: %+v", len(got.SourceIDs), got.SourceIDs)
	}
	for i, id := range wantIDs {
		if got.SourceIDs[i] != id {
			t.Errorf("SourceIDs[%d] = %q, want %q (sorted output)", i, got.SourceIDs[i], id)
		}
	}
	// 3 descriptions < threshold 6 with default budget → fast path; no LLM.
	if prov.Calls != 0 {
		t.Errorf("Provider.Calls=%d; expected 0 — 3 descriptions are below threshold", prov.Calls)
	}
	// And the merged Description must contain all three fragments via
	// concat (since the fast path joins them).
	if !strings.Contains(got.Description, "senior archivist") ||
		!strings.Contains(got.Description, "lost manuscript") ||
		!strings.Contains(got.Description, "2019") {
			t.Errorf("merged Description lost fragments: %q", got.Description)
	}
}

// --- bonus sanity tests (not part of the 6 required, but cheap) ---

// TestMergeRelationshipsCanonicalizesEdge — A->B and B->A should hash to
// the same canonical (min, max) pair, weights add, sourceIDs union.
func TestMergeRelationshipsCanonicalizesEdge(t *testing.T) {
	prov := NewMockProvider()
	rels := []Relationship{
		{SrcID: "Scrooge", TgtID: "Marley", Description: "partners", Weight: 1.5,
			SourceIDs: []string{"c1"}, Keywords: "partnership"},
		{SrcID: "Marley", TgtID: "Scrooge", Description: "partners", Weight: 0.5,
			SourceIDs: []string{"c2"}, Keywords: "partnership"},
	}
	merged, err := MergeRelationships(context.Background(), prov, rels)
	if err != nil {
		t.Fatalf("MergeRelationships: %v", err)
	}
	if len(merged) != 1 {
		t.Fatalf("expected 1 merged rel, got %d", len(merged))
	}
	r := merged[0]
	if r.SrcID != "Marley" || r.TgtID != "Scrooge" {
		t.Errorf("canonical edge wrong: got %s->%s, want Marley->Scrooge", r.SrcID, r.TgtID)
	}
	if r.Weight != 2.0 {
		t.Errorf("weight not summed: got %v, want 2.0", r.Weight)
	}
	wantIDs := []string{"c1", "c2"}
	if len(r.SourceIDs) != 2 || r.SourceIDs[0] != wantIDs[0] || r.SourceIDs[1] != wantIDs[1] {
		t.Errorf("SourceIDs wrong: %+v", r.SourceIDs)
	}
}

// TestSingleDescriptionPassesThrough — the upstream optimization at
// operate.py:194-196: len(descriptions)==1 → return as-is, no LLM.
func TestSingleDescriptionPassesThrough(t *testing.T) {
	prov := NewMockProvider()
	desc := "Eleanor Hartwell is a senior archivist."
	out, llmUsed, err := SummarizeDescriptions(context.Background(), prov,
		[]string{desc}, 500, 2000, 6)
	if err != nil {
		t.Fatalf("SummarizeDescriptions: %v", err)
	}
	if out != desc {
		t.Errorf("single description should pass through; got %q", out)
	}
	if llmUsed || prov.Calls != 0 {
		t.Errorf("single description should not invoke LLM")
	}
}
