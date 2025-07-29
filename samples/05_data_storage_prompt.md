### **Prompt for `05_data_storage.md`**

**Scenario:** A large media company is building a new analytics platform. They have vast amounts of unstructured (video, audio) and semi-structured (logs, social media feeds) data, along with some structured customer data. They need a scalable, cost-effective storage solution that supports both batch processing for historical analysis and interactive querying for data scientists. Data lifecycle management and cost optimization are key concerns.

**Task:** As an LLM-powered Storage Architect, using the provided `05_data_storage.md` document as your primary reference, design a multi-layered data storage architecture for this media company.

**Specific Requirements:**
1.  **Storage Layer Design:** Propose a multi-layered storage architecture (e.g., Bronze, Silver, Gold) based on the "Decision Tree: Storage Layer Design" (Section 2.1). Describe the purpose and typical data characteristics for each layer in this scenario.
2.  **Technology Selection:** For each proposed layer, recommend specific storage technologies (e.g., S3, ADLS, Redshift, BigQuery, Delta Lake) from the "Storage Technology Selection Matrix" (Section 2.2) that are best suited for the data types and access patterns. Justify your choices.
3.  **Data Lake Architecture:** Outline how the "Medallion Architecture" (Section 4.1) would be implemented for the unstructured and semi-structured data, detailing the role of the Bronze, Silver, and Gold layers.
4.  **Lifecycle Management:** Describe a data lifecycle management strategy for the raw unstructured data, referencing the "S3 Intelligent Tiering Configuration" (Section 3.1) or similar concepts for other cloud providers. Include rules for transitioning data to colder storage tiers.
5.  **Data Organization:** Explain how data partitioning and file formats (e.g., Parquet, Delta) would be used to optimize query performance and cost, drawing from the "Bronze Layer Configuration" (Section 4.1) and general best practices.

**Constraint:** Your response must explicitly reference the sections and concepts from the `05_data_storage.md` document that informed your decisions, alongside your general knowledge of industry best practices.
