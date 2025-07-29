### **Prompt for `06_data_processing.md`**

**Scenario:** Your company has a large data lake containing raw and semi-processed data from various sources. They need to perform complex transformations, aggregations, and machine learning feature engineering on this data to prepare it for analytical dashboards and predictive models. The processing needs vary from daily batch jobs to ad-hoc interactive analysis.

**Task:** As an LLM-powered Data Engineer, using the provided `06_data_processing.md` document as your primary reference, design a data processing strategy that can handle diverse workloads and optimize for performance and cost.

**Specific Requirements:**
1.  **Processing Pattern:** Based on the "Processing Pattern Decision Framework" (Section 2.1), recommend a suitable processing pattern (e.g., Batch, Stream, Interactive) for the described scenario, considering both daily jobs and ad-hoc analysis.
2.  **Processing Engine Selection:** Recommend a primary data processing engine from the "Processing Engine Selection Matrix" (Section 2.2) that can handle complex transformations and feature engineering on a large data lake. Justify your choice based on scalability, flexibility, and ecosystem.
3.  **Transformation Framework:** Outline a high-level transformation framework, referencing the "Transformation Frameworks" (Section 3.1). Describe how you would structure the code for reusability and maintainability (e.g., using modular functions, dataframes).
4.  **Performance Optimization:** Suggest at least two performance optimization techniques from the "Performance Optimization Guide" (Section 6) that would be crucial for large-scale data processing, such as partitioning, caching, or efficient file formats.
5.  **Orchestration:** Briefly describe how these processing jobs would be orchestrated, referencing the "Orchestration Tools" (Section 4.1) and considering the need for scheduling daily jobs and enabling ad-hoc execution.
6.  **Error Handling:** Outline a strategy for handling errors during data processing, referencing the "Troubleshooting & Diagnostics" (Section 8.1), including mechanisms for logging, alerting, and data reprocessing.

**Constraint:** Your response must explicitly reference the sections and concepts from the `06_data_processing.md` document that informed your decisions, alongside your general knowledge of industry best practices.
