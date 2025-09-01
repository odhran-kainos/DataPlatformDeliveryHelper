---
title: SDLC Phase Index
version: 1.0.0
last_reviewed: 2025-01-01
framework_version: 0.1.0
tags: ["index", "navigation", "phases"]
---

# Data Platform SDLC Phase Index

## Overview

This index provides a comprehensive view of all phases in the Data Platform Software Development Life Cycle, their current status, key outputs, and related prompt patterns.

## Phase Summary Table

| ID | Phase Name | Purpose | Key Outputs | Related Prompts | Status |
|----|------------|---------|-------------|-----------------|--------|
| 00 | [SDLC Overview](00_data_platform_sdlc.md) | High-level lifecycle framework and guiding principles | Framework overview, phase relationships | [Framework Orientation](../sample_prompts/00_framework_orientation_prompt.md) | active |
| 01 | [Discovery & Requirements](01_discovery_requirements.md) | Capture business needs and establish project foundation | Requirements document, stakeholder matrix, success criteria | [Discovery Requirements](../sample_prompts/01_discovery_requirements_prompt.md) | active |
| 02 | [Architecture & Design](02_architecture_design.md) | Define technical architecture and system design | Architecture diagrams, technology selection, design documents | [Architecture Design](../sample_prompts/02_architecture_design_prompt.md) | active |
| 03 | [Environment Setup](03_environment_setup.md) | Establish development and deployment environments | Infrastructure as code, CI/CD pipelines, environment configs | [Environment Setup](../sample_prompts/03_environment_setup_prompt.md) | active |
| 04 | [Data Ingestion](04_data_ingestion.md) | Design and implement data ingestion capabilities | Ingestion pipelines, connectors, data validation rules | [Data Ingestion](../sample_prompts/04_data_ingestion_prompt.md) | active |
| 05 | [Data Storage](05_data_storage.md) | Implement data storage layers and management | Storage architecture, partitioning strategy, data catalog | [Data Storage](../sample_prompts/05_data_storage_prompt.md) | active |
| 06 | [Data Processing](06_data_processing.md) | Build data transformation and processing workflows | ETL/ELT pipelines, business logic, processing schedules | [Data Processing](../sample_prompts/06_data_processing_prompt.md) | active |
| 07 | [Analytics & Consumption](07_analytics_consumption.md) | Enable data consumption and analytics capabilities | APIs, dashboards, reporting solutions, data marts | [Analytics Consumption](../sample_prompts/07_analytics_consumption_prompt.md) | active |
| 08 | [Testing & Quality](08_testing_quality.md) | Implement testing frameworks and quality assurance | Test suites, data quality rules, validation frameworks | [Testing Quality](../sample_prompts/08_testing_quality_prompt.md) | active |
| 09 | [Deployment & Release](09_deployment_release.md) | Deploy to production and manage releases | Deployment scripts, release procedures, rollback plans | [Deployment Release](../sample_prompts/09_deployment_release_prompt.md) | active |
| 10 | [Monitoring & Support](10_monitoring_support.md) | Establish operational monitoring and support | Monitoring dashboards, alerting rules, support procedures | [Monitoring Support](../sample_prompts/10_monitoring_support_prompt.md) | active |
| 11 | [Documentation & Knowledge](11_documentation_knowledge.md) | Create and maintain comprehensive documentation | Technical documentation, user guides, knowledge base | [Documentation Knowledge](../sample_prompts/11_documentation_knowledge_prompt.md) | active |
| 12 | [Continuous Improvement](12_continuous_improvement.md) | Optimize and evolve the data platform | Performance optimizations, feature enhancements, lessons learned | [Continuous Improvement](../sample_prompts/12_continuous_improvement_prompt.md) | active |

## Advanced Topic Supplements

| ID | Topic Name | Purpose | Primary Phase | Status |
|----|------------|---------|---------------|--------|
| 05b | [Advanced Data Storage](05b_data_storage_advanced.md) | Deep dive into complex storage patterns and optimization | 05 - Data Storage | active |
| 06b | [Advanced Data Processing](06b_data_processing_advanced.md) | Complex processing patterns and performance optimization | 06 - Data Processing | active |
| 07b | [Advanced Analytics](07b_analytics_consumption_advanced.md) | Advanced analytics patterns and real-time processing | 07 - Analytics & Consumption | active |

## Navigation Guide

### For New Users
1. Start with [00_data_platform_sdlc.md](00_data_platform_sdlc.md) for framework overview
2. Review [QUICK_START.md](../QUICK_START.md) for immediate value
3. Follow phases sequentially for comprehensive understanding

### For Specific Needs
- **Requirements gathering**: Phase 01
- **Technology selection**: Phase 02
- **Implementation guidance**: Phases 03-09
- **Operations setup**: Phases 10-11
- **Optimization**: Phase 12

### For LLM Integration
- **Context preparation**: Use phase documents as LLM context
- **Prompt selection**: Choose corresponding prompt from sample_prompts directory
- **Quality validation**: Apply [EVALUATION_RUBRIC.md](../EVALUATION_RUBRIC.md)
- **Responsible usage**: Follow [RESPONSIBLE_AI.md](../RESPONSIBLE_AI.md)

## Status Definitions

- **active**: Document is complete, reviewed, and ready for use
- **draft**: Document is in development, may have incomplete sections
- **deprecated**: Document is being phased out, use alternative
- **planned**: Document is planned for future development

## Contributing to Phase Documentation

See [CONTRIBUTING.md](../CONTRIBUTING.md) for:
- Documentation standards and front matter requirements
- Template usage guidelines ([_TEMPLATE_PHASE.md](_TEMPLATE_PHASE.md))
- Review and validation processes
- Community contribution guidelines

## Framework Evolution

This index is updated as the framework evolves. Check [CHANGELOG.md](../CHANGELOG.md) for recent changes and [METRICS.md](../METRICS.md) for framework effectiveness measurements.