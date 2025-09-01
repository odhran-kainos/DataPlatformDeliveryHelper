---
title: Contributing Guidelines
version: 1.0.0
last_reviewed: 2025-01-01
framework_version: 0.1.0
---

# Contributing to Data Platform SDLC Framework

## Purpose

This repository serves as an LLM-optimized knowledge base for data platform development, providing structured documentation, prompt patterns, and evaluation assets that enable AI-assisted software delivery lifecycle management.

## Repository Scope

This framework provides:
- **LLM-optimized SDLC knowledge base**: Structured documentation for data platform development phases
- **Prompt patterns**: Reusable templates and examples for effective LLM interactions
- **Evaluation assets**: Rubrics, metrics, and validation frameworks for assessing AI-generated outputs
- **Governance artifacts**: Guidelines for responsible AI use, contribution standards, and quality assurance

## How to Propose Changes

1. **Create an Issue**: Before making changes, create an issue describing the proposed enhancement or fix
2. **Fork and Branch**: Fork the repository and create a feature branch from `main`
3. **Follow Standards**: Ensure all changes adhere to the documentation authoring standards below
4. **Submit PR**: Submit a pull request with clear description linking to the related issue
5. **Review Process**: Changes will be reviewed for consistency, accuracy, and alignment with framework goals

## Documentation Authoring Standards

### File Naming Conventions
- **Phase documents**: `NN_phase_name.md` (e.g., `01_discovery_requirements.md`)
- **Template files**: `_TEMPLATE_*.md` (e.g., `_TEMPLATE_PHASE.md`)
- **Supporting docs**: Use descriptive names in `UPPER_CASE.md` format

### Front Matter Specification
All documentation files must include YAML front matter with these keys:
```yaml
---
phase_id: "01"                    # Phase identifier (for phase docs)
title: "Document Title"
status: "active|draft"            # Document status
version: "1.0.0"                  # Semantic version
tags: ["tag1", "tag2"]           # Relevant tags
owners: ["contributor1"]          # Document maintainers
last_reviewed: "YYYY-MM-DD"      # Last review date
framework_version: "0.1.0"       # Framework version compatibility
---
```

### Length and Structure Guidance
- **Abstracts**: Keep introductions under 180 words
- **Front matter**: Limit to 15 lines maximum
- **Content**: Favor structured lists and tables over long prose blocks
- **Sections**: Use imperative headings (e.g., "Define Requirements" vs "Requirements Definition")

### Writing Style
- **Token efficiency**: Write concisely for optimal LLM consumption
- **Imperative headings**: Use action-oriented section titles
- **Lists over prose**: Structure information as bulleted or numbered lists
- **Tables for decisions**: Use decision matrices and comparison tables
- **Code blocks**: Use four backticks for markdown file blocks within repository docs

### Prompt Files

Prompt files follow the naming convention: `NN_phase_name_prompt.md`

Required fields for all prompt files:
```yaml
---
goal: "Specific objective of this prompt"
context_requirements: ["required document", "additional context"]
expected_output_format: "JSON|Markdown|Table"
phase_id: "01"
version: "1.0.0"
---
```

## Validation Checklist

Before submitting changes, ensure:
- [ ] **Spell check**: All content is spell-checked and grammatically correct
- [ ] **Front matter**: All required front matter fields are present and valid
- [ ] **References**: All internal links and references are resolved and working
- [ ] **Format consistency**: Document follows established formatting patterns
- [ ] **Schema validation**: JSON schemas pass basic linting (for schema files)
- [ ] **Token optimization**: Content is structured for efficient LLM consumption

## Review Process

1. **Automated checks**: Basic validation of format and links
2. **Content review**: Assessment of technical accuracy and completeness
3. **Framework alignment**: Verification that changes support LLM optimization goals
4. **Community feedback**: Open review period for stakeholder input

## Questions or Issues?

For questions about contributing, please:
1. Check existing issues for similar questions
2. Create a new issue with the "question" label
3. Provide context about your proposed contribution

Thank you for helping improve the Data Platform SDLC Framework!