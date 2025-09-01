---
title: Quick Start Guide
version: 1.0.0
last_reviewed: 2025-01-01
framework_version: 0.1.0
tags: ["quickstart", "getting-started", "tutorial"]
---

# Quick Start Guide

## Goal

Get immediate value from the Data Platform SDLC Framework by leveraging LLMs to accelerate your data platform development process in under 30 minutes.

## Prerequisites

- **LLM Access**: ChatGPT, Claude, or similar large language model with context window ≥8K tokens
- **Retrieval Optional**: RAG system setup not required for basic usage
- **Project Context**: Existing or planned data platform initiative

## 5-Step Fast Path

### Step 1: Discovery & Requirements (5 minutes)
1. Copy content from `docs/01_discovery_requirements.md`
2. Use `sample_prompts/01_discovery_requirements_prompt.md` as your prompt template
3. Get structured requirements and stakeholder analysis

### Step 2: Data Ingestion Planning (5 minutes)
1. Copy content from `docs/04_data_ingestion.md`
2. Use `sample_prompts/04_data_ingestion_prompt.md`
3. Generate ingestion architecture and implementation plan

### Step 3: Data Modeling & Storage (10 minutes)
1. Copy content from `docs/05_data_storage.md`
2. Use `sample_prompts/05_data_storage_prompt.md`
3. Design storage layers and data organization strategy

### Step 4: Quality & Testing Framework (5 minutes)
1. Copy content from `docs/08_testing_quality.md`
2. Use `sample_prompts/08_testing_quality_prompt.md`
3. Establish data quality rules and testing approach

### Step 5: Deployment Review (5 minutes)
1. Copy content from `docs/09_deployment_release.md`
2. Use `sample_prompts/09_deployment_release_prompt.md`
3. Create deployment checklist and release strategy

## Example Session

Here's a copy-paste example for immediate testing:

### Context (Copy this to your LLM):
```
You are a Data Platform Delivery Assistant. I'm providing you with structured documentation about data platform development phases. Use this context to provide detailed, actionable guidance.

[PASTE CONTENT FROM docs/01_discovery_requirements.md HERE]
```

### Prompt (Then submit this):
```
Scenario: Our e-commerce company needs a data platform to "understand customers better and improve sales."

Task: Using the provided discovery framework, create:
1. Stakeholder identification matrix with 5 key groups
2. Three effective requirements gathering techniques for e-commerce
3. Five critical workshop questions to uncover business needs
4. Key deliverables list for this discovery phase
5. Three measurable success metrics

Output: Structured response with explicit references to document sections that informed your recommendations.
```

### Expected Output:
- Structured stakeholder analysis
- Requirements gathering plan
- Workshop question bank
- Deliverables checklist
- Success metrics framework

## Next Steps

### Deeper Documentation Exploration
- **Phase-by-phase**: Work through `docs/00_data_platform_sdlc.md` for complete lifecycle
- **Advanced topics**: Explore `docs/0Xb_*_advanced.md` files for deep technical guidance
- **Cross-references**: Follow internal links between related phases

### Evaluation & Quality
- **Use rubric**: Apply `EVALUATION_RUBRIC.md` to assess LLM outputs
- **Check metrics**: Track effectiveness using `METRICS.md` framework
- **Responsible AI**: Review `RESPONSIBLE_AI.md` for ethical guidelines

### Customization
- **Modify prompts**: Adapt sample prompts for your specific technology stack
- **Create templates**: Use `docs/_TEMPLATE_PHASE.md` for custom phase documentation
- **Schema validation**: Leverage `OUTPUT_SCHEMAS/` for structured outputs

### Community & Contribution
- **Issues**: Report problems or suggest improvements via GitHub issues
- **Contribute**: Follow `CONTRIBUTING.md` to add new patterns or documentation
- **Share**: Document your successful prompt patterns for community benefit

## Troubleshooting

**LLM doesn't reference document sections**: Ensure you're providing the full document content as context and explicitly requesting section references in your prompt.

**Output lacks specificity**: Add more constraints and examples to your prompts, following patterns in `sample_prompts/prompt_patterns.md`.

**Hallucinated information**: Apply validation checklist from `RESPONSIBLE_AI.md` and cross-reference outputs with source documentation.

Ready to accelerate your data platform development with AI assistance!