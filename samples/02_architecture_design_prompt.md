### **Prompt for `02_architecture_design.md`**

**Scenario:** Following a successful discovery phase, your team has identified key requirements for a new data platform: it needs to support both batch analytics for historical reporting and real-time dashboards for operational insights. The company is cloud-agnostic but prefers open-source technologies where possible.

**Task:** As an LLM-powered Data Architect, using the provided `02_architecture_design.md` document as your primary reference, design a high-level data platform architecture that meets these requirements.

**Specific Requirements:**
1.  **Architectural Pattern:** Recommend a suitable architectural pattern (e.g., Lambda, Kappa, Data Mesh) from the "Architectural Patterns Decision Framework" (Section 2.1) that can accommodate both batch and real-time processing. Justify your choice.
2.  **Core Components:** Identify the core architectural components for each layer (Ingestion, Storage, Processing, Consumption) of your chosen pattern, referencing the "Core Architectural Components" (Section 3.1). For each component, suggest a generic open-source technology (e.g., Kafka, Spark, MinIO) that fits the cloud-agnostic preference.
3.  **Data Flow Diagram:** Describe the high-level data flow through your proposed architecture, explaining how data moves from source to consumption for both batch and real-time paths.
4.  **Technology Selection Matrix:** For a specific component (e.g., real-time processing), use the "Technology Selection Matrix" (Section 2.2) to justify your choice of open-source technology over a proprietary cloud service, considering factors like scalability, cost, and operational complexity.
5.  **Security Integration:** Briefly outline how security considerations would be integrated into this architecture, referencing the "Security Integration Patterns" (Section 4.1).

**Constraint:** Your response must explicitly reference the sections and concepts from the `02_architecture_design.md` document that informed your decisions, alongside your general knowledge of industry best practices.
