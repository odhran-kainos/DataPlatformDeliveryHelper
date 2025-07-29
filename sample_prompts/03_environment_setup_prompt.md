### **Prompt for `03_environment_setup.md`**

**Scenario:** Your company is expanding its data platform to a new region (e.g., Europe) and needs to provision a new production environment. They are considering both AWS and Azure for this expansion, but have limited cloud expertise in-house. Cost optimization and security are paramount.

**Task:** As an LLM-powered Infrastructure Architect, using the provided `03_environment_setup.md` document as your primary reference, design a comprehensive production environment setup.

**Specific Requirements:**
1.  **Cloud Provider Recommendation:** Based on the "Environment Tier Strategy Matrix" and "Infrastructure-as-Code Tool Selection" (Section 2), recommend the most suitable cloud provider (AWS or Azure) for a *regulated enterprise* with a focus on cost and security, given limited in-house cloud expertise. Justify your choice.
2.  **Infrastructure-as-Code (IaC) Tool Selection:** Recommend a specific IaC tool (e.g., Terraform, CloudFormation, Bicep) from the document that aligns with your cloud provider recommendation and the company's need for cost optimization and security. Explain why this tool is a good fit.
3.  **Environment Configuration Template:** Generate a JSON configuration snippet for the `production` environment, drawing from the "Environment Variables Template" (Section 3.2). Ensure it reflects best practices for a production environment (e.g., resource sizing, auto-scaling, backup retention, monitoring, security, cost optimization).
4.  **Security Baseline:** Outline key security configurations for this production environment, referencing the "AWS Security Configuration" or "Azure Security Configuration" (Section 4.1) as applicable to your chosen cloud. Include aspects like encryption, network security, access control, and monitoring.
5.  **CI/CD Integration:** Briefly describe how this environment provisioning would integrate into a CI/CD pipeline, referencing the "Data Platform Deployment Pipeline" (Section 5.1) or "Azure DevOps Pipeline Template" (Section 5.2). Highlight critical steps like validation, security scanning, and manual approval for production.

**Constraint:** Your response must explicitly reference the sections and concepts from the `03_environment_setup.md` document that informed your decisions, alongside your general knowledge of industry best practices.
