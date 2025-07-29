### **Prompt for `07_analytics_consumption.md`**

**Scenario:** Your company has a well-established data platform with a curated Gold layer. Now, various business units need to consume this data for different purposes: executive dashboards, self-service analytics for business users, and data science exploration. The challenge is to provide flexible and performant access while maintaining data governance and security.

**Task:** As an LLM-powered Analytics Architect, using the provided `07_analytics_consumption.md` document as your primary reference, design a comprehensive data consumption strategy that caters to diverse user needs.

**Specific Requirements:**
1.  **Consumption Pattern:** Based on the "Consumption Pattern Decision Framework" (Section 2.1), recommend suitable consumption patterns for each user group (executives, business users, data scientists). Justify your choices.
2.  **Consumption Tool Selection:** For each recommended pattern, suggest appropriate consumption tools from the "Consumption Tool Selection Matrix" (Section 2.2) (e.g., Power BI, Tableau, Jupyter, SQL clients). Explain why these tools are a good fit for their respective user groups.
3.  **Data Modeling:** Briefly describe how data modeling (e.g., star schema, data vault) would be applied to the Gold layer to optimize for analytical consumption, referencing the "Data Modeling Best Practices" (Section 3.1).
4.  **Performance Optimization:** Suggest at least two performance optimization techniques from the "Performance Optimization Guide" (Section 6) that would improve query performance for analytical consumption (e.g., caching, materialized views, indexing).
5.  **Data Governance:** Outline how data governance principles (e.g., access control, data cataloging, data lineage) would be applied to ensure secure and compliant data consumption, referencing the "Data Governance Framework" (Section 4.1).
6.  **Monitoring:** Identify key metrics for monitoring data consumption and user activity, drawing from the "Monitoring & Observability" (Section 7.1).

**Constraint:** Your response must explicitly reference the sections and concepts from the `07_analytics_consumption.md` document that informed your decisions, alongside your general knowledge of industry best practices.
