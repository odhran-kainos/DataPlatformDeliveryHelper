### **Prompt for `04_data_ingestion.md`**

**Scenario:** A financial services company needs to build a new data ingestion pipeline for real-time transaction data from various sources. The data is highly sensitive and requires strict data quality checks and robust error handling. They are currently using Apache Kafka for event streaming and want to integrate this new pipeline into their existing ecosystem.

**Task:** As an LLM-powered Data Engineer, using the provided `04_data_ingestion.md` document as your primary reference, design a real-time data ingestion pipeline for this scenario.

**Specific Requirements:**
1.  **Ingestion Pattern:** Based on the "Ingestion Pattern Selection Decision Tree" (Section 2.1), identify the most appropriate ingestion pattern for real-time, high-volume transaction data.
2.  **Streaming Connector:** Recommend a specific streaming connector from the "Streaming Connectors" section (Section 3.3) that integrates with Apache Kafka. Provide a high-level Python code snippet for a Kafka consumer, demonstrating how it would consume messages and include basic error handling.
3.  **Data Quality Rules:** Define at least three critical data quality rules for transaction data (e.g., `transaction_id`, `amount`, `timestamp`) using the "Data Quality Engine" concepts (Section 5.1). Specify the `check_type`, `columns`, and `severity` for each.
4.  **Error Handling Strategy:** Detail a robust error handling strategy for the pipeline, referencing the "Troubleshooting & Diagnostics" section (Section 8.1). Include mechanisms for retries, dead-letter queues, and alerting for critical failures.
5.  **Performance Optimization:** Suggest at least two performance optimization strategies from the "Performance Optimization Guide" (Section 6) to ensure the pipeline can handle high throughput and low latency requirements.
6.  **Monitoring Metrics:** Identify key metrics for monitoring this real-time pipeline, drawing from the "Custom Metrics Collection" (Section 7.1).

**Constraint:** Your response must explicitly reference the sections and concepts from the `04_data_ingestion.md` document that informed your decisions, alongside your general knowledge of industry best practices.
