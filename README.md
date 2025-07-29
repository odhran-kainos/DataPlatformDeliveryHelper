# Data Platform SDLC Framework (LLM-Optimized)

This repository contains a comprehensive framework for designing, developing, and managing data platforms, optimized for use with Large Language Models (LLMs) as a contextual knowledge base. The primary goal is to enable LLMs to assist in or even automate the creation and management of data platform components by providing them with structured, actionable documentation.

## Project Structure

```
.
├── docs/
│   ├── 00_data_platform_sdlc.md
│   ├── 01_discovery_requirements.md
│   └── ... (all other SDLC documents)
├── sample_prompts/
│   ├── 01_discovery_requirements_prompt.md
│   └── ... (all other sample prompts)
├── Review Of Version 1.md
├── README.md
└── .gitignore
```

*   **`docs/`**: This directory contains the core documentation for the Data Platform SDLC (Software Development Life Cycle). Each markdown file (`00_data_platform_sdlc.md` to `12_continuous_improvement.md`, including advanced topics like `05b_data_storage_advanced.md`) covers a specific phase or aspect of data platform development. These documents are structured with LLM consumption in mind, featuring decision trees, matrices, code templates, and best practices.
*   **`sample_prompts/`**: This directory contains example prompts designed to demonstrate how to effectively query an LLM using the SDLC documents as context. Each prompt is tailored to a specific SDLC document and aims to elicit detailed, actionable responses from the LLM, showcasing its ability to synthesize information from the provided context and its own general knowledge.
*   **`Review Of Version 1.md`**: This file contains a review or summary of the initial version of the project.
*   **`README.md`**: This file provides an overview of the repository, its purpose, and how to use its contents.
*   **`.gitignore`**: Specifies intentionally untracked files that Git should ignore.

## How to Use the SDLC Documents

The markdown files within the `docs/` directory are designed to serve as a rich, structured knowledge base for LLMs.

1.  **Contextual Input:** When interacting with an LLM (e.g., ChatGPT, Claude, custom LLM deployments), provide the relevant SDLC document(s) as part of the context. This can be done by:
    *   Copy-pasting the content of the markdown file directly into the LLM's prompt.
    *   Using an LLM API that supports context windows large enough to accommodate the document content.
    *   Integrating the documents into a Retrieval-Augmented Generation (RAG) system.
2.  **Actionable Guidance:** The documents contain decision frameworks, tool comparisons, code snippets, and best practices. LLMs can leverage this information to:
    *   Generate infrastructure-as-code.
    *   Design data pipelines.
    *   Recommend technologies.
    *   Troubleshoot issues.
    *   Outline strategies for various data platform challenges.

## How to Use the Sample Prompts

The prompts in the `sample_prompts/` directory are ready-to-use examples to test and demonstrate the effectiveness of the SDLC documents.

1.  **Select a Prompt:** Choose a prompt from the `sample_prompts/` directory that corresponds to the SDLC document you want to test (e.g., `sample_prompts/04_data_ingestion_prompt.md` for `docs/04_data_ingestion.md`).
2.  **Provide Context:** Copy the content of the corresponding SDLC document (e.g., `docs/04_data_ingestion.md`) and provide it to your LLM as context.
3.  **Submit the Prompt:** Copy the content of the chosen sample prompt and submit it to the LLM.
4.  **Analyze Response:** Observe how the LLM synthesizes information from the provided document and its own knowledge to generate a detailed and relevant response. The prompts are designed to encourage the LLM to explicitly reference sections and concepts from the document.

This framework aims to streamline the data platform development process by making expert knowledge readily available and actionable through LLM assistance.
