# Contributing Guide (Internal Only)

> **External contributions are NOT accepted.** Do **not** share any part of this repository outside Kainos Software Ltd. (No clients, vendors, contractors, public repos, presentations, or training decks.) Employees only.

Thank you for helping improve the Internal Data Platform SDLC Framework (LLM‑Optimised). This guide defines how internal employees contribute while preserving consistency, confidentiality, and quality.

## Purpose
Provide a structured, machine-friendly body of knowledge and reusable prompt patterns to accelerate delivery of modern data platforms.

## Repository Scope
- Structured SDLC phase documents (strategic + actionable guidance)
- Prompt patterns and phase-specific sample prompts
- Output schemas for generated artefacts (validation targets)
- Governance, evaluation, metrics, responsible AI guidelines
- (Future) Automation scripts for retrieval, evaluation, and CI linting

Out of scope (for now): Production infrastructure modules, proprietary vendor benchmarks, externally distributed tooling.

## Internal Change Workflow
1. Raise an internal ticket / issue describing the change.
2. (Optional) Discuss / design outline if substantial.
3. Create a branch + PR referencing the issue (small, logically scoped).
4. Complete the validation checklist.
5. Request review (≥1 reviewer; 2 for governance / schema changes).
6. Address comments; squash or tidy commits as needed.
7. Merge following approval.

## SDLC Document Authoring Standards
- File naming: `NN_phase_name.md` (zero‑padded).
- One phase per file; large appendices may be split (`01_discovery_requirements_appendix_tools.md`).
- Abstract (≤ 180 words) must be high signal and retrieval‑friendly.
- Prefer lists, tables, structured headings over long prose.
- Avoid vendor lock‑in language unless a comparison matrix explicitly calls it out.
- Include required front matter (see below) and a Revision History table.
- Cite internal references or public standards (no proprietary third-party text).

## Front Matter Specification (Docs)
```yaml
phase_id: "01"
title: "Discovery & Requirements"
status: draft          # draft | active | deprecated
version: 0.1.0
framework_version: 0.1.0
tags: [discovery, requirements]
owners: ["odhran-kainos"]
last_reviewed: 2025-09-01
next_review_due: 2025-12-01
```
Optional keys: `maturity`, `dependencies`, `aliases`.

## Front Matter (Prompt Files)
```yaml
phase_id: "01"
type: prompt
status: draft
goal: "Elicit structured requirements baseline"
expected_output_format: json+markdown
```

## Writing Style
- Use imperative headings: Define, Capture, Evaluate, Decide.
- Concise sentences; remove filler.
- Consistent terminology (e.g. “data product”).
- Canonical data quality dimension order: completeness, accuracy, timeliness, consistency, uniqueness.
- Mark placeholders with `<<placeholder>>`.

## Prompt File Standards
Include: Goal, When to Use, Required Context, Prompt Template (parameterised), Output Specification, Validation Steps, Failure / refinement loop hints.

## Output Schemas
- Live in `OUTPUT_SCHEMAS/`.
- JSON Schema Draft 2020‑12.
- Provide `title`, `description`, `type`, `required`.
- snake_case property names.
- Supply an example (in referencing doc) for critical schemas.

## Validation Checklist (Before PR)
- [ ] Front matter present & valid
- [ ] Abstract ≤ 180 words
- [ ] Decision matrices use tables
- [ ] Internal relative links resolve
- [ ] Spell / lint pass
- [ ] Referenced schemas exist
- [ ] Prompts specify explicit output format
- [ ] CHANGELOG updated if framework-visible change
- [ ] No confidential third-party content introduced

## Versioning
- Framework semantic version tracked in root CHANGELOG.
- Individual doc `version` increments on substantive content changes.
- Deprecated docs retained with status: deprecated until replaced.

## Reviews
- ≥1 reviewer for ordinary docs; 2 for governance, responsible AI, or schema changes.
- Use structured comments referencing headings or line ranges.

## Responsible AI Alignment
Follow RESPONSIBLE_AI.md for:
- Data sensitivity & redaction
- Model selection (latency, cost, quality, compliance trade‑offs)
- Verification / traceability habits
- Hallucination mitigation patterns
- Ethical & IP considerations
- Audit trail recommendations

## Metrics (Initial)
Qualitative focus (clarity, adoption, decision velocity). Planned quantitative instrumentation (future):
- Time-to-draft-phase-artifact
- Defect / rework rate in generated artefacts
- Phase coverage (% phases with active prompt patterns)
- Schema conformance pass rate
- Prompt success score (rubric average)

## Confidentiality & IP
All content is proprietary & confidential (see LICENSE). Do not copy outside internal secure systems. Avoid introducing:
- Third-party code / text with restrictive or viral licences
- Sensitive client data / secrets
- Personal data beyond minimal attribution in owners lists

## Licence / Rights
All contributions become part of the proprietary internal knowledge base owned by Kainos Software Ltd. No external redistribution rights are granted. By contributing you confirm you have the right to submit the material and that it does not include externally restricted content.

## Questions
Open an internal issue with label `question`. External discussions are not permitted.
```