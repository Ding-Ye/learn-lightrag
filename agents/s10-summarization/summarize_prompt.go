package main

// summarize_prompt.go holds the system/user prompt used by SummarizeDescriptions
// when the heuristic decides an LLM merge is required. The text below is a
// FAITHFUL SUMMARY of upstream's PROMPTS["summarize_entity_descriptions"]
// (lightrag/prompt.py:185-218), NOT a verbatim copy. We preserve every
// load-bearing instruction (role, JSONL input format, third-person
// objectivity, conflict handling, length constraint, language preservation)
// and use the SAME placeholder names so callers can format the prompt with
// fmt.Sprintf-equivalent string-replacement on:
//
//   {description_type}   "entity" or "relation"
//   {description_name}   the canonical entity/relation name
//   {description_list}   the JSONL-formatted descriptions block
//   {summary_length}     soft token budget (e.g. 500)
//   {language}           output language (e.g. "English")
//
// Bumping summaryPromptVersion invalidates downstream caches that key on
// (chunk||promptVersion) — change it whenever you edit summaryPrompt below.
//
// Upstream cite (verified 2026-05-09):
//   lightrag/prompt.py:185      PROMPTS["summarize_entity_descriptions"] start
//   lightrag/prompt.py:203      length constraint clause
//   lightrag/prompt.py:204-207  language clause
//   lightrag/operate.py:319-330 prompt_template.format(...) call site

const summaryPromptVersion = "v1"

// summaryPrompt is the system+user merged template.  Placeholders use the
// upstream {brace} convention so the formatRender() helper in summarize.go
// can do simple string-replace.  We drop the upstream's "Output:" label
// because the LLM convention reliably emits the summary text immediately
// after the prompt without it.
const summaryPrompt = `---Role---
You are a Knowledge Graph Specialist, proficient in data curation and synthesis.

---Task---
Synthesize a list of descriptions of a given entity or relation into one
comprehensive, cohesive summary.

---Instructions---
1. Input format: the Description List below is JSONL — one JSON object per
   line, each with a "Description" field.
2. Output format: plain text, in multiple paragraphs if needed; no preamble,
   no commentary, no bullet markers.
3. Comprehensiveness: integrate every key fact from every input description.
   Do not omit details.
4. Objectivity: write in third-person, explicitly naming the entity or
   relation at the start so the summary stands alone without extra context.
5. Conflict handling:
   - If conflicting descriptions look like two distinct entities sharing
     a name, summarize each separately within the same output.
   - If they describe one entity with disagreement, reconcile when possible
     or present both viewpoints with noted uncertainty.
6. Length constraint: total length must not exceed {summary_length} tokens
   while maintaining depth.
7. Language: write the output in {language}.  Proper nouns may stay in
   their original language if no widely accepted translation exists.

---Input---
{description_type} Name: {description_name}

Description List:

` + "```" + `
{description_list}
` + "```" + `

(Upstream cite: lightrag/prompt.py:185-218, summarized.  The full upstream
prompt also enumerates 8 sections vs our 7 — we collapsed the duplicate
"Context" / "Context & Objectivity" headings into a single Objectivity rule.)`
