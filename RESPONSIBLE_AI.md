---
title: Responsible AI Guidelines
version: 1.0.0
last_reviewed: 2025-01-01
framework_version: 0.1.0
tags: ["responsible-ai", "ethics", "governance", "validation"]
---

# Responsible AI Guidelines

## Scope

These guidelines ensure ethical, reliable, and secure usage of Large Language Models (LLMs) within the Data Platform SDLC Framework. They apply to all AI-assisted activities including documentation generation, code creation, architecture design, and decision support.

## Data Sensitivity & Redaction

### Classification Levels
- **Public**: Open source documentation, general best practices
- **Internal**: Company-specific architecture patterns, non-sensitive metadata
- **Confidential**: Customer data schemas, security configurations, proprietary algorithms
- **Restricted**: Personal data, credentials, compliance-sensitive information

### Redaction Requirements
Before sharing context with LLMs:
- [ ] **Remove credentials**: API keys, passwords, connection strings
- [ ] **Anonymize data**: Replace real customer names, addresses, sensitive identifiers
- [ ] **Generalize specifics**: Replace proprietary system names with generic equivalents
- [ ] **Mask compliance data**: Redact regulatory or legal sensitive information

### Safe Context Patterns
```
✅ Good: "Design ingestion for retail transaction data with 1M records/day"
❌ Avoid: "Design ingestion for CustomerDB.transactions with John Smith's credit card data"
```

## Verification Checklist

Apply this checklist to all LLM-generated outputs before implementation:

### Factual Accuracy
- [ ] **Technology capabilities**: Verify technical claims against official documentation
- [ ] **Version compatibility**: Check software versions and API compatibility
- [ ] **Performance claims**: Validate throughput, latency, and scalability assertions
- [ ] **Security features**: Confirm security capabilities and limitations

### Decision Validation
- [ ] **Business alignment**: Ensure recommendations align with stated requirements
- [ ] **Cost implications**: Verify cost estimates and optimization suggestions
- [ ] **Risk assessment**: Validate identified risks and proposed mitigations
- [ ] **Implementation feasibility**: Confirm technical implementability

### Compliance Verification
- [ ] **Regulatory requirements**: Check against GDPR, CCPA, industry regulations
- [ ] **Security standards**: Validate against organization security policies
- [ ] **Data governance**: Ensure alignment with data management standards
- [ ] **Audit requirements**: Verify traceability and logging recommendations

## Hallucination Mitigation Patterns

### Context Grounding Techniques
1. **Explicit source references**: Require LLM to cite specific document sections
2. **Constraint specification**: Provide clear boundaries and limitations
3. **Example-driven prompts**: Include concrete examples of expected outputs
4. **Validation requests**: Ask LLM to identify assumptions and uncertainties

### Prompt Anti-Patterns to Avoid

#### Over-broad Requests
```
❌ "Design a complete data platform"
✅ "Design data ingestion layer using provided requirements and constraints"
```

#### Implicit Dependencies
```
❌ "Recommend the best storage solution"
✅ "Recommend storage solution for 100TB analytical workload with <1s query latency"
```

#### Unconstrained Generation
```
❌ "Generate monitoring strategy"
✅ "Generate monitoring strategy using metrics from section 4.2, focusing on data quality"
```

### Validation Patterns
- **Contradiction check**: Ask LLM to identify potential conflicts in recommendations
- **Alternative analysis**: Request multiple approaches with trade-off analysis
- **Assumption listing**: Explicitly request assumptions underlying recommendations
- **Risk identification**: Ask for failure modes and edge cases

## Ethical Considerations

### Bias Awareness
- **Technology bias**: Recognize LLM preferences for certain technology stacks
- **Recency bias**: Account for training data cutoff dates in technology recommendations
- **Complexity bias**: Be aware of tendency toward over-engineered solutions
- **Scale bias**: Consider appropriateness for different organization sizes

### Intellectual Property
- **Code generation**: Verify generated code doesn't violate licenses or copyrights
- **Architecture patterns**: Ensure recommendations don't infringe on proprietary designs
- **Attribution**: Credit sources when adapting existing patterns or frameworks
- **Open source compliance**: Validate license compatibility for recommended tools

### Human Oversight
- **Critical decisions**: Require human review for architecture, security, and compliance decisions
- **Production deployment**: Never deploy LLM-generated code without testing and review
- **Stakeholder input**: Ensure human stakeholders validate business requirements and priorities
- **Expert consultation**: Engage domain experts for specialized technical areas

## Audit Trail Suggestions

### Documentation Requirements
- **Prompt versioning**: Maintain versions of prompts used for critical decisions
- **Context tracking**: Document source materials provided to LLM
- **Output validation**: Record verification steps and any corrections made
- **Decision rationale**: Capture reasoning behind adopting or rejecting LLM recommendations

### Traceability Patterns
```yaml
llm_session:
  date: "2025-01-01"
  model: "gpt-4"
  context_docs: ["01_discovery_requirements.md", "04_data_ingestion.md"]
  prompt_version: "1.2.0"
  validation_checklist: "completed"
  human_reviewer: "architect@company.com"
  modifications: ["Updated security controls", "Added compliance requirements"]
```

### Review Cycles
- **Weekly review**: Check recent LLM-assisted decisions for accuracy and alignment
- **Monthly audit**: Assess patterns in LLM usage and effectiveness
- **Quarterly evaluation**: Review and update responsible AI guidelines based on learnings
- **Annual assessment**: Comprehensive evaluation of AI integration maturity and risks

## Implementation Guidelines

### Team Training
- Provide training on responsible AI principles and these specific guidelines
- Establish clear escalation paths for ethical concerns or validation questions
- Create communities of practice for sharing responsible AI patterns and lessons learned

### Tool Integration
- Integrate validation checklists into development workflows
- Automate redaction patterns where possible
- Implement approval gates for critical AI-assisted decisions

### Continuous Improvement
- Collect feedback on guideline effectiveness and usability
- Update guidelines based on new AI capabilities and risks
- Share learnings with broader community while respecting confidentiality

Remember: AI is a powerful assistant, but human judgment, domain expertise, and ethical oversight remain essential for responsible data platform development.