# Contributing Guide

Thank you for your interest in improving the Data Platform SDLC Framework (LLM‑Optimized). This guide explains how to propose changes and maintain consistency across the documentation and prompt assets.

## Purpose
Provide a structured, machine-friendly body of knowledge and reusable prompt patterns to accelerate delivery of modern data platforms.

## Repository Scope
- Structured SDLC phase documents (strategic + actionable guidance)
- Prompt patterns and phase-specific sample prompts
- Output schemas for generated artefacts (validation targets)
- Governance, evaluation, metrics, responsible AI guidelines
- (Future) Automation scripts for retrieval, evaluation, and CI linting

Out of scope (for now): Full application code, proprietary vendor benchmarks, production infrastructure modules.

## How to Propose Changes
1. Open an issue describing the change (or enhancement / bug / doc gap).
2. Discuss if clarification is needed.
3. Open a PR referencing the issue. Keep PRs logically scoped.
4. Ensure all validation checklist items pass (see below).
5. Request review.

## SDLC Document Authoring Standards
- File naming: `NN_phase_name.md` (zero‑padded, sequential).
- One phase per file; deep appendices may be split (e.g. `01_discovery_requirements_appendix_tools.md`).
- Keep abstract (first 180 words) high-value and scannable (acts as chunk priority in retrieval).
- Prefer lists, tables, structured headings over narrative prose.
- Avoid vendor lock‑in language unless explicitly in a comparison matrix.
- Include **front matter** (see below) and a Revision History section.
- Cite sources or internal references when making prescriptive recommendations.

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
Additional optional keys: `maturity`, `dependencies`, `aliases`.

## Front Matter (Prompt Files)
```yaml
phase_id: "01"
type: prompt
status: draft
goal: "Elicit structured requirements baseline"
expected_output_format: json+markdown
```

## Writing Style
- Use imperative heading verbs: Define, Capture, Evaluate, Decide
- Keep sentences concise; remove filler ("very", "highly", etc.)
- Normalize terminology ("data product", not alternating synonyms)
- Prefer canonical enumeration order for repeated lists (e.g. completeness, accuracy, timeliness, consistency, uniqueness)
- Mark placeholders clearly with `<<placeholder>>`

## Prompt File Standards
- File naming: `NN_phase_name_prompt.md` (or `prompt_patterns.md` for catalogues)
- Include: Goal, When to Use, Required Context, Prompt Template (parameterized), Output Specification, Validation Steps
- Encourage idempotent / reconstructible outputs (explicit schema or markdown section requirements)

## Output Schemas
Schemas live in `OUTPUT_SCHEMAS/`. If adding a new schema:
1. Use JSON Schema Draft 2020‑12.
2. Provide `title`, `description`, `type`, `required`.
3. Keep property names snake_case.
4. Add example in a fenced code block within referencing docs.

## Validation Checklist (Before Opening / Updating a PR)
- [ ] Front matter present & valid keys
- [ ] Abstract <= 180 words
- [ ] Tables used for decision matrices
- [ ] Internal relative links resolve
- [ ] Spell check / lint pass
- [ ] Output schemas referenced exist
- [ ] Prompts specify explicit output format
- [ ] CHANGELOG updated (if user-facing change)

## Versioning
- Framework version in root CHANGELOG.
- Each phase doc `version` increments when substantive content changes (not typos).
- Deprecated docs retain file (status: deprecated) until replaced.

## Reviews
- At least one reviewer for docs; two for governance or schema changes.
- Encourage structured review comments referencing line numbers or section heading.

## License Alignment
Contributions are licensed under the repository MIT License. Do not contribute proprietary or confidential material.

## Questions
Open an issue with the label `question` or start a discussion (future addition).