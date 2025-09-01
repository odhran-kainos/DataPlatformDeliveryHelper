---
title: Reusable Prompt Patterns Library
version: 1.0.0
last_reviewed: 2025-01-01
framework_version: 0.1.0
tags: ["prompts", "patterns", "templates", "library"]
---

# Reusable Prompt Pattern Library

## Overview

This library provides modular prompt components that can be combined and customized for effective LLM interactions within the Data Platform SDLC framework. Use these patterns to create consistent, high-quality prompts across different phases and use cases.

## Core Prompt Components

### Context Preamble
Standard opening that establishes the LLM's role and context:

```
You are a Data Platform Delivery Assistant with expertise in enterprise data architecture, engineering best practices, and modern data technologies. I'm providing you with structured documentation about data platform development phases. Use this context to provide detailed, actionable guidance.

Context provided:
- [List specific documents or sections provided]
- [Additional context materials]

Instructions:
- Reference specific document sections when making recommendations
- Provide concrete, implementable solutions
- Consider enterprise constraints and best practices
- Identify potential risks and mitigation strategies
```

### Task Statement Pattern
Clear, actionable task definition:

```
Task: [Specific objective]

Scenario: [Business context and constraints]

Requirements:
1. [Specific requirement 1]
2. [Specific requirement 2]
3. [Specific requirement 3]

Context constraints:
- Technology preferences: [Technology stack or preferences]
- Scale requirements: [Data volume, user count, etc.]
- Timeline: [Project timeline constraints]
- Budget considerations: [Cost constraints or optimization goals]
```

### Output Format Specification
Structured output requirements:

```
Output Format:
Provide your response in the following structure:

1. **Executive Summary** (2-3 sentences)
2. **Detailed Analysis**
   - Key findings
   - Recommendations
   - Implementation approach
3. **Decision Matrix** (if applicable)
   - Options comparison table
   - Criteria weights and scores
4. **Implementation Plan**
   - Step-by-step approach
   - Timeline estimates
   - Resource requirements
5. **Risk Assessment**
   - Identified risks
   - Mitigation strategies
6. **Next Steps** (prioritized action items)

Format: Use markdown with clear headings and bullet points for optimal readability.
```

### Validation Request Pattern
Quality assurance and verification instructions:

```
Validation Requirements:
- Cite specific sections from provided documentation that support your recommendations
- Identify any assumptions you're making and mark them clearly
- Highlight areas where additional information would improve the recommendation
- Consider alternative approaches and briefly explain why you chose your recommended approach
- Flag any potential conflicts between requirements or constraints
```

### Critique Loop Pattern
Self-assessment and improvement mechanism:

```
Self-Assessment:
After providing your initial response, perform a brief self-critique:

1. **Completeness Check**: Are all requirements addressed?
2. **Practicality Assessment**: Is this implementable given the constraints?
3. **Risk Evaluation**: Have I identified the major risks?
4. **Alternative Consideration**: Are there other viable approaches I should mention?
5. **Clarity Review**: Is this guidance clear and actionable?

If you identify any gaps in your initial response, provide additional clarification or alternative recommendations.
```

## Specialized Pattern Examples

### 1. Decomposition Pattern
For breaking down complex problems:

```
Decomposition Task:
Break down [complex problem] into manageable components:

1. **Component Identification**
   - List 3-5 major components
   - Define clear boundaries between components
   - Identify dependencies and interfaces

2. **Prioritization**
   - Rank components by implementation priority
   - Consider dependencies and risk factors
   - Suggest parallel vs. sequential implementation

3. **Detailed Planning**
   - For each component, provide:
     - Effort estimate
     - Key risks
     - Success criteria
     - Required expertise

Reference framework sections that support your decomposition approach.
```

### 2. Risk Assessment Pattern
For comprehensive risk analysis:

```
Risk Assessment Framework:
Analyze [scenario/decision] using this structured approach:

1. **Risk Identification**
   - Technical risks (scalability, performance, integration)
   - Business risks (timeline, budget, stakeholder)
   - Operational risks (maintenance, support, skills)
   - Compliance risks (security, regulatory, privacy)

2. **Risk Quantification**
   | Risk | Probability | Impact | Risk Score | Mitigation Cost |
   |------|-------------|--------|------------|-----------------|
   | Risk 1 | H/M/L | H/M/L | H/M/L | High/Med/Low |

3. **Mitigation Strategies**
   - Preventive measures
   - Contingency plans
   - Risk monitoring approaches

Cite relevant risk management sections from the provided documentation.
```

### 3. Schema Generation Pattern
For creating structured outputs:

```
Schema Generation Task:
Create a structured schema for [data type/output format]:

Requirements:
- Follow JSON Schema Draft 2020-12 standard
- Include all required fields based on use case analysis
- Provide clear descriptions for each field
- Define appropriate data types and constraints
- Include examples for complex fields

Validation:
- Ensure schema supports all identified use cases
- Verify required vs. optional field classifications
- Check for proper validation rules and constraints
- Consider extensibility for future requirements

Reference any existing schemas in the OUTPUT_SCHEMAS directory for consistency.
```

### 4. Decision Justification Pattern
For explaining choices and trade-offs:

```
Decision Justification Framework:
For [decision being made], provide comprehensive justification:

1. **Options Considered**
   - List 3-5 viable alternatives
   - Brief description of each option

2. **Evaluation Criteria**
   - Define 5-7 key criteria with weights
   - Explain why these criteria are important

3. **Scoring Matrix**
   | Option | Criteria 1 | Criteria 2 | Criteria 3 | Total Score |
   |--------|------------|------------|------------|-------------|
   | A      | 8/10       | 6/10       | 9/10       | 7.7/10      |

4. **Recommendation Rationale**
   - Why the recommended option is best
   - Key advantages and trade-offs
   - Implementation considerations

5. **Sensitivity Analysis**
   - How robust is this decision to changing requirements?
   - What factors could change the recommendation?

Support your analysis with references to decision frameworks in the provided documentation.
```

## Prompt Quality Guidelines

### Effective Prompt Characteristics
- **Specific**: Clear, measurable objectives
- **Contextual**: Relevant business and technical context
- **Structured**: Organized output requirements
- **Validatable**: Criteria for assessing response quality
- **Actionable**: Results that can be directly implemented

### Common Prompt Anti-Patterns

#### Vague Objectives
```
❌ "Help me design a data platform"
✅ "Design a real-time data ingestion architecture for 100K events/second with <5 second latency SLA"
```

#### Missing Context
```
❌ "What's the best database for our use case?"
✅ "Given our 10TB analytical workload with complex joins and BI tool integration, recommend database options using criteria from Section 5.2"
```

#### Unstructured Output
```
❌ "Give me recommendations"
✅ "Provide recommendations in a decision matrix format with criteria weights, scores, and implementation timelines"
```

#### No Validation Criteria
```
❌ "Is this architecture good?"
✅ "Evaluate this architecture against the validation framework in Section 7.1, scoring each dimension 0-3"
```

## Integration with Framework

### Phase-Specific Patterns
- Each phase document has corresponding prompt patterns optimized for that phase's objectives
- Use phase-specific patterns as starting points, then customize with general patterns
- Combine multiple patterns for complex, multi-objective prompts

### Quality Assurance
- Apply the evaluation rubric from [EVALUATION_RUBRIC.md](../EVALUATION_RUBRIC.md) to assess prompt effectiveness
- Use responsible AI guidelines from [RESPONSIBLE_AI.md](../RESPONSIBLE_AI.md) for ethical considerations
- Track prompt performance using metrics from [METRICS.md](../METRICS.md)

### Continuous Improvement
- Document successful prompt variations for community sharing
- Analyze prompt effectiveness patterns and update library accordingly
- Contribute new patterns based on emerging use cases and technologies

---

*This pattern library is designed to evolve. Contribute successful patterns and improvements following the guidelines in [CONTRIBUTING.md](../CONTRIBUTING.md).*