# PROJECT SYSTEM AUDIT: AI Data Analyzer

**Date:** February 18, 2026  
**Status:** Phase 26 Complete  
**Document Purpose:** Comprehensive capability, architecture, and dependency mapping for strategic product planning.

---

## 1. APPLICATION OVERVIEW

- **Project Name:** AI Data Analyzer (Analyzer)
- **Short Description:** An agentic data intelligence pipeline that transforms raw CSV data into executive-grade strategic reports.
- **Primary Purpose:** Automate the end-to-end data analysis lifecycle—from ingestion and cleaning to forecasting and professional reporting.
- **Target User Type:** Analysts, Business Executives, and Operations Managers.
- **Core Value Proposition:** Reduction of "Time to Insight" from hours to seconds by automating data hygiene and high-level strategic reasoning.

---

## 2. ARCHITECTURE OVERVIEW

### Structure
The system follows a **Modular Agentic Architecture** orchestrated by a central master controller. Each agent is a specialized unit of logic with a strict contract (Input -> Execution -> Output).

### Orchestrator Role
The `MasterOrchestrator` manages the sequential execution of the pipeline, handles state transitions, collects logs/metrics, and ensures data integrity between stages.

### Pipeline Flow (Step-by-Step)
1.  **Ingestion**: Raw CSV is parsed and schema types are detected.
2.  **Quality Audit (Before)**: Baseline data quality score is calculated.
3.  **Cleaning**: Technical hygiene (missing values, duplicates) is applied.
4.  **Repair**: Semantic repair reasoning treats noise while maintaining statistical truth.
5.  **Quality Audit (After)**: Post-cleaning improvement is verified.
6.  **Insights**: Trends, correlations, and anomalies are detected; LLM generates strategic narrative.
7.  **Forecasting**: Time-series suitability is detected and future projections are modeled.
8.  **Dashboarding**: Dynamic HTML visualization suite is generated.
9.  **Reporting**: Final PDF is rendered using "Executive Visual Framing" (Page 1 strategy, high-density grids).

### Layer Definitions
-   **Frontend**: Streamlit-based interface for interactive exploration and file uploads.
-   **Backend API**: FastAPI-powered engine for headless pipeline execution.
-   **Validation Layer**: Multi-stage validation preventing "hallucinated" or invalid data from reaching reports.
-   **Forecast Layer**: Scipy-powered linear regression and scenario simulation engine.
-   **NL Query Layer**: LLM-driven natural language interface for ad-hoc data interrogation.

---

## 3. COMPLETE FEATURE LIST

- **Data Handling**
    - High-speed CSV ingestion and auto-schema detection.
    - Automated cleaning (Deduplication, Standard Null Handling).
    - Repair Reasoning: Semantic preservation during data correction.
- **Analytics & Intelligence**
    - 24-point Data Quality Scoring (Before/After).
    - 2-layer Statistical Trend Detection (Linear Regression + Significance).
    - Multi-factor Outlier/Anomaly Detection (Z-Score > 3).
    - Relationship Intensity Matrix (Correlations > 0.7).
    - Non-Obvious Signal Filtering (Derived column detection).
- **Projections & Simulation**
    - Automated Time-Series Detection.
    - 10-period Linear Forecasting with 95% Confidence Intervals.
    - "What-If" Scenario Simulation (Metric factor adjustment).
- **User Interface & Output**
    - NL Query Engine: "Chat with your data" to generate custom charts.
    - Dynamic Dashboard: Interactive Plotly suite with metric drill-down.
    - Executive PDF Reporting: High-density 2x3 KPI grids, Risk Badges, and "Business Implication" callouts.
    - High-Resolution Chart Embedding (Exported at 2x scale).

---

## 4. AGENT INVENTORY

| Agent | Responsibility | Input | Output |
| :--- | :--- | :--- | :--- |
| **IngestionAgent** | File parsing & Schema detection | CSV File | `IngestionResult` |
| **CleaningAgent** | Technical hygiene & Normalization | `IngestionResult` | `CleaningResult` |
| **RepairReasoningAgent** | Semantic repair logic | `CleaningResult` | `RepairResult` |
| **DataQualityAgent** | Scoring & Validation metrics | DataFrame | `DataQualityResult` |
| **InsightAgent** | Statistical & Strategic Analysis | `RepairResult` | `InsightResult` |
| **ForecastAgent** | Linear projections & Scenarios | `InsightResult` | `ForecastResult` |
| **NLQueryAgent** | Natural language interrogation | Query String | `NLQueryResult` |
| **ReportValidationEngine**| Logic Guardrails | `Insight|Forecast` | Validated Result |
| **ReportAgent** | Visual Framing & PDF Generation | All Pipeline Data | PDF/MD Report |

---

## 5. DEPENDENCY & INTEGRATION LIST

- **Streamlit**: Primary frontend for the interactive user dashboard.
- **FastAPI**: Backend infrastructure for API and microservice capability.
- **Plotly**: Generation of interactive and high-resolution static charts.
- **ReportLab**: PDF engine for multi-page flowable document generation.
- **Pandas / Numpy**: Core data manipulation and matrix mathematics.
- **Scipy / Scikit-learn**: Statistical modeling, regression, and outlier detection.
- **Gemini / OpenAI API**: Strategic reasoning, strategic narrative, and NL Query interpretation.

---

## 6. REQUIRED CREDENTIALS / API KEYS

- **LLM_API_KEY**: Required for Strategic Insights and NL Query (can be OpenAI or Gemini).
- **LLM_PROVIDER**: Choice of `openai`, `gemini`, or `ollama`.
- **LLM_MODEL**: Model target (e.g., `gemini-1.5-pro` or `gpt-4o-mini`).
- **LLM_ENDPOINT**: API routing (Standard or Proxy).
- **JWT_SECRET**: Required for API authentication (Enterprise mode).

---

## 7. ENVIRONMENT CONFIGURATION

- **Python Version**: 3.10+ recommended.
- **Core Packages**: `fastapi`, `uvicorn`, `pandas`, `numpy`, `plotly`, `streamlit`, `reportlab`.
- **Port Configuration**:
    - **API**: 8000
    - **Frontend (Streamlit)**: 8501
- **Deployment Options**: Dockerized (Dockerfile and docker-compose.yml available).

---

## 8. DATA LIMITATIONS

- **Forecast Threshold**: Requires minimum **8 data points** for high-quality projections (Agent allows 5, Sanitization requires 8).
- **Correlation Ceiling**: Identity or redundant correlations (abs > 0.999) are suppressed to avoid trivial insights.
- **Trend Threshold**: p-value < 0.05 required for significance; slope < 0.0001 filtered as "Stable".
- **File Size**: Default limit is **50MB** (Configurable).
- **Memory**: Operation is strictly in-memory (Pandas); large datasets (>1M rows) may require scaling.

---

## 9. VALIDATION RULES

- **Null Handling**: NaN values are replaced with "-" in KPIs and ignored in statistical calculations.
- **Derived Logic**: Columns with a ratio standard deviation < 0.01 are flagged as "Derived" and suppressed from relationship matrices.
- **Forecast Quality**: Forecasts with $R^2 < 0.1$ are suppressed from executive reports to prevent misleading signals.

---

## 10. CURRENT WEAKNESSES

- **Visual Gaps**: Current PDF relies on standard Helvetica; lacks support for complex layout layering (e.g., overlapping elements).
- **Forecasting Sophistication**: Limited to Linear Regression; lacks seasonality (Prophet/ARIMA) and multi-variable forecasting.
- **Performance**: PDF generation includes chart rendering which can spike latency during batch operations.
- **Technical Debt**: Internal analytics agents perform repetitive statistical checks; could be centralized into a shared Math provider.
- **Scalability**: Limited by single-node horizontal scaling; lacks distributed task queue (Celery).

---

## 11. FUTURE EXPANSION OPPORTUNITIES

- **Intelligence**: Integration of multi-variable causal modeling and root-cause analysis.
- **Financial Modeling**: Dedicated profit/loss simulation and ROI calculation engines.
- **Enterprise Features**: SSO integration, multi-tenant workspace management, and Slack/Teams alerting.
- **UX Improvements**: Drag-and-drop report layout builder and "Voice to Insight" capability.
- **Monetization**: Usage-based billing (tokens/reports) and White-labeled PDF reporting.

---

## 12. CONFIDENCE & MATURITY ASSESSMENT

- **System Maturity Score**: 8/10 (High stability in core pipeline).
- **Production Readiness**: High (Validated against dirty data, high-res reporting active).
- **Sellability**: High (Clear executive output and high-density KPI visuals).
- **Risk Areas**: LLM dependency for strategic logic; prompt drift.
