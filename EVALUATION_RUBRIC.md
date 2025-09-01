---
title: Evaluation Rubric for LLM Outputs
version: 1.0.0
last_reviewed: 2025-01-01
framework_version: 0.1.0
tags: ["evaluation", "quality", "rubric", "assessment"]
---

# Evaluation Rubric for LLM-Generated Outputs

## Overview

This rubric provides a structured framework for evaluating the quality and effectiveness of LLM-generated content within the Data Platform SDLC context. Use this for assessing documentation, code, architecture recommendations, and other AI-assisted deliverables.

## Scoring Scale

**Scale**: 0-3 points per dimension
- **0**: Unacceptable - Major deficiencies requiring complete rework
- **1**: Below Standard - Significant issues requiring substantial revision
- **2**: Meets Standard - Acceptable quality with minor improvements needed
- **3**: Exceeds Standard - High quality output requiring minimal or no revision

## Evaluation Dimensions

### 1. Relevance (Weight: 20%)

**Assesses**: How well the output addresses the specific prompt requirements and business context.

| Score | Criteria | Examples |
|-------|----------|----------|
| 3 | Directly addresses all prompt requirements with clear connection to business goals | Architecture recommendation perfectly aligned with stated scalability and compliance needs |
| 2 | Addresses most requirements with good business context understanding | Minor gaps in requirement coverage but generally appropriate |
| 1 | Partially addresses requirements with some business context awareness | Generic recommendations with loose connection to specific needs |
| 0 | Fails to address key requirements or lacks business context understanding | Off-topic response or ignores critical constraints |

### 2. Completeness (Weight: 15%)

**Assesses**: Whether the output covers all necessary aspects and provides sufficient detail for implementation.

| Score | Criteria | Examples |
|-------|----------|----------|
| 3 | Comprehensive coverage of all relevant aspects with appropriate detail level | Complete data pipeline design including error handling, monitoring, and security |
| 2 | Covers most important aspects with adequate detail | Good coverage but missing some secondary considerations |
| 1 | Covers basic aspects but lacks detail or misses important elements | High-level overview without implementation specifics |
| 0 | Significant gaps in coverage or insufficient detail for practical use | Incomplete solution missing critical components |

### 3. Actionability (Weight: 20%)

**Assesses**: How easily the output can be translated into concrete implementation steps.

| Score | Criteria | Examples |
|-------|----------|----------|
| 3 | Provides clear, specific implementation steps with concrete examples | Step-by-step deployment guide with code samples and configuration examples |
| 2 | Generally actionable with some specific guidance | Implementation approach with most steps defined |
| 1 | Provides direction but requires significant interpretation for implementation | High-level guidance requiring substantial additional planning |
| 0 | Too abstract or vague for practical implementation | Conceptual overview without actionable steps |

### 4. Specificity (Weight: 15%)

**Assesses**: Level of detail and precision in recommendations, avoiding generic or vague guidance.

| Score | Criteria | Examples |
|-------|----------|----------|
| 3 | Highly specific recommendations with precise parameters and configurations | "Use Apache Kafka with 3 brokers, replication factor 3, batch.size=16384 for high-throughput ingestion" |
| 2 | Good level of specificity with some precise recommendations | Technology choices with basic configuration guidance |
| 1 | Some specific elements but includes generic recommendations | Mix of specific and vague guidance |
| 0 | Predominantly generic or vague recommendations | "Use appropriate tools" or "follow best practices" without specifics |

### 5. Traceability (Weight: 15%)

**Assesses**: How well the output references and builds upon provided documentation sources.

| Score | Criteria | Examples |
|-------|----------|----------|
| 3 | Explicitly cites specific document sections and builds logically on provided context | "Following the ingestion patterns in Section 4.2, implementing batch processing with the decision matrix from Table 3.1" |
| 2 | Good use of provided context with some explicit references | Uses framework concepts with occasional source citations |
| 1 | Some connection to provided materials but limited explicit referencing | General alignment with framework but minimal source attribution |
| 0 | Little to no connection to provided documentation context | Ignores provided context or contradicts source materials |

### 6. Risk Awareness (Weight: 10%)

**Assesses**: Recognition and appropriate handling of potential risks, limitations, and trade-offs.

| Score | Criteria | Examples |
|-------|----------|----------|
| 3 | Proactively identifies risks with specific mitigation strategies | Identifies scalability risks and provides specific architectural patterns to address them |
| 2 | Recognizes important risks with general mitigation approaches | Acknowledges security concerns with standard protection recommendations |
| 1 | Limited risk identification or generic mitigation suggestions | Mentions some risks but without specific solutions |
| 0 | Fails to identify significant risks or provides inadequate mitigation | Overlooks critical security, performance, or compliance risks |

### 7. Output Conformance (Weight: 5%)

**Assesses**: Adherence to requested output format, schema, or structural requirements.

| Score | Criteria | Examples |
|-------|----------|----------|
| 3 | Perfect adherence to requested format with clear structure | JSON output matching exact schema with all required fields |
| 2 | Good format compliance with minor deviations | Follows format with small structural variations |
| 1 | Partial format compliance requiring some restructuring | Generally correct format but missing some elements |
| 0 | Poor format compliance requiring significant restructuring | Wrong format or missing critical structural elements |

## Scoring Matrix

### Total Score Calculation
```
Total Score = (Relevance × 0.20) + (Completeness × 0.15) + (Actionability × 0.20) + 
              (Specificity × 0.15) + (Traceability × 0.15) + (Risk Awareness × 0.10) + 
              (Output Conformance × 0.05)
```

### Quality Interpretation

| Total Score | Quality Level | Recommended Action |
|-------------|---------------|-------------------|
| 2.7 - 3.0 | Excellent | Use with minimal review, consider as exemplar |
| 2.3 - 2.6 | Good | Use with light review and minor adjustments |
| 1.8 - 2.2 | Acceptable | Requires moderate revision before use |
| 1.0 - 1.7 | Poor | Substantial rework needed, consider re-prompting |
| 0.0 - 0.9 | Unacceptable | Discard and restart with improved prompt |

## Usage Guidelines

### When to Apply
- Before implementing LLM-generated architectures or code
- During quality reviews of AI-assisted documentation
- When evaluating prompt effectiveness for continuous improvement
- For training team members on quality standards

### Evaluation Process
1. **Initial Assessment**: Quick review for obvious quality issues
2. **Detailed Scoring**: Apply rubric systematically across all dimensions
3. **Gap Analysis**: Identify specific areas needing improvement
4. **Improvement Actions**: Revise output or refine prompts based on gaps
5. **Learning Capture**: Document patterns for future prompt optimization

### Team Calibration
- Conduct team scoring exercises on sample outputs
- Discuss scoring rationale to align on quality standards
- Regular calibration sessions to maintain consistency
- Create organization-specific scoring examples and edge cases

## Common Quality Patterns

### High-Quality Indicators
- Specific technology recommendations with version numbers and configuration details
- Explicit references to provided documentation sections
- Proactive identification of implementation challenges and solutions
- Clear step-by-step implementation guidance
- Appropriate consideration of organizational constraints

### Quality Red Flags
- Generic "best practices" without specific implementation guidance
- Recommendations that ignore provided constraints or requirements
- Missing consideration of security, compliance, or scalability concerns
- Output that doesn't reference provided context materials
- Overly complex solutions without justification for complexity

Remember: This rubric is a tool for continuous improvement, not a gate for perfection. Use it to identify patterns and enhance prompt engineering effectiveness over time.