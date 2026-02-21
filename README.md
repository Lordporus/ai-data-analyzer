# AI Data Analyzer


# Take a Look
https://ai-data-analyst-59zk.onrender.com/

**A Deterministic-First, Multi-Agent Data Intelligence System**

---

## Overview

Most data analysis workflows remain manual, repetitive, and inconsistent. Analysts spend disproportionate time on tasks that should be automated: parsing file schemas, handling missing values, detecting column types, computing summary statistics, building charts, and formatting reports. The outputs vary depending on who performs the work and when.

AI Data Analyzer eliminates this friction. It is a modular, multi-agent analytics pipeline that takes a raw CSV file as input and produces a complete analytical deliverable: cleaned data, statistical insights, time-series forecasts, an interactive dashboard, and an executive-grade PDF report. The system handles every step from ingestion to final output without requiring manual intervention.

The system is built for reliability. Every core computation is deterministic. There are no required API keys, no paid service dependencies, and no opaque inference steps. Outputs are reproducible, auditable, and consistent across runs. LLM-powered features exist as optional enhancements that degrade gracefully when unavailable.

This project is designed for data analysts, engineering teams evaluating automated pipelines, and portfolio reviewers looking for evidence of production-grade systems thinking.

---

## Architecture Overview

The system follows a modular multi-agent design. Each agent is a self-contained processing unit with a single responsibility, a typed input, and a typed output. Agents do not share state. The orchestrator invokes them in strict sequence, collects their results, and assembles a unified pipeline output consumed by both the frontend dashboard and the report generator.

### Pipeline Flow

```
CSV Upload
    |
    v
IngestionAgent         Schema detection, type inference, metadata extraction
    |
    v
CleaningAgent          Null imputation, duplicate removal, type coercion
    |
    v
RepairAgent            Structural normalization, column renaming, encoding fixes
    |
    v
DataQualityAgent       Quality scoring across completeness, consistency, validity
    |
    v
InsightAgent           KPIs, trend analysis, correlation matrix, risk detection
    |
    v
ForecastAgent          Time-series forecasting with adaptive model selection
    |
    v
ReportValidationEngine Statistical filtering, NaN sanitization, derived column suppression
    |
    v
ReportAgent            Executive PDF report generation
```

### Design Principles

The architecture is governed by three rules:

1. **Deterministic by default.** Every agent produces the same output for the same input. No stochastic steps exist in the critical path.
2. **AI as augmentation, not dependency.** LLM features are confined to optional layers (natural language querying, narrative enhancement). The system produces complete output without them.
3. **Guardrail-based reporting.** A dedicated validation engine filters every insight before it reaches the report. Weak statistical findings are suppressed rather than presented.

---

## Core Features

### Data Handling

The ingestion pipeline detects column types using strict matching thresholds, handles mixed-type columns, and normalizes datetime formats. The cleaning agent imputes missing numeric values with column medians and categorical values with column modes. Duplicates are removed deterministically. All transformations are logged for auditability.

### Insight Engine

The InsightAgent computes key performance indicators, linear trend analysis with R-squared and p-value metrics, pairwise correlation matrices, and risk/opportunity classification. Insights are deterministic and computed using scipy and numpy. No LLM is involved in statistical analysis.

### Forecasting (Adaptive)

The forecast engine implements adaptive model selection between linear regression and Holt-Winters exponential smoothing based on detected data characteristics. See the dedicated section below for technical details.

### Executive Reporting

The ReportAgent generates PDF reports using ReportLab Platypus flowables. Reports follow a consulting-grade structure with metric-aware narrative commentary that adapts to the type of metric being discussed (volume, efficiency, dispersion). Raw statistical values are translated into business language.

### Validation and Guardrails

The ReportValidationEngine acts as a quality gate between analysis and presentation. It suppresses derived columns, filters trivial correlations (r > 0.999), drops forecasts with NaN values, and enforces minimum data thresholds. This prevents misleading or statistically weak findings from appearing in final outputs.

### Optional AI Enhancements

When configured with an LLM API key, the system enables two additional features:

- Natural language querying of uploaded datasets
- Enhanced narrative generation in report summaries

Both features are optional. If the LLM call fails or is unavailable, the system falls back to template-based output with no functional loss.

---

## Forecasting Engine

The ForecastAgent implements a two-tier model selection strategy designed for reliability over complexity.

### Model Selection

The engine evaluates each numeric time-series column and selects a model based on the data's statistical profile:

- If autocorrelation analysis confirms a seasonal pattern (correlation > 0.5 at any lag, dominant lag >= 2, and sufficient data length), the engine uses Holt-Winters exponential smoothing with additive trend and seasonality.
- Otherwise, it defaults to linear regression with confidence intervals derived from residual standard deviation.

This is a deliberate design choice. Complex models applied to insufficient data produce unreliable projections. The engine prefers a well-supported simple model over a poorly-supported complex one.

### Confidence Grading

Each forecast receives a confidence grade based on R-squared, p-value, and volatility index:

| Grade | Criteria |
|:------|:---------|
| High | R-squared > 0.7, p-value < 0.05, low volatility |
| Moderate | R-squared 0.4 to 0.7, mixed statistical support |
| Low | R-squared < 0.4 or high residual variance |

### Suppression

If fewer than 8 historical data points exist, the forecast is not generated. This threshold is enforced at both the ForecastAgent and the ReportValidationEngine. Forecasts containing NaN values are also dropped before reaching the report layer.

---

## Running the Project

### Prerequisites

- Python 3.10 or later
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Start the Interactive Dashboard

```bash
streamlit run frontend/app.py --server.port 8501
```

The dashboard will be accessible at `http://localhost:8501`.

### Start the API Server (Optional)

```bash
uvicorn api.main:app --reload --port 8000
```

The API will be accessible at `http://localhost:8000`. API documentation is auto-generated at `/docs`.

---

## Environment Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

The system works without any environment variables set. All core features (ingestion, cleaning, analysis, forecasting, reporting, dashboard) function in fully deterministic mode without API keys.

To enable optional LLM features, configure the following in `.env`:

```
LLM_PROVIDER=gemini
LLM_API_KEY=your-api-key
LLM_MODEL=gemini-flash-latest
LLM_ENDPOINT=https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
```

If these are not set, the system operates in deterministic mode with no degradation in core functionality.

---

## Project Structure

```
ai-data-analyzer/
├── agents/                  Core agent implementations
│   ├── base.py              Base agent interface
│   ├── ingestion.py         Schema detection and type inference
│   ├── cleaning.py          Data cleaning and null handling
│   ├── repair.py            Structural repair and normalization
│   ├── data_quality.py      Quality scoring engine
│   ├── insight.py           Statistical analysis and KPI computation
│   ├── forecast.py          Adaptive forecasting engine
│   ├── report_validation.py Validation quality gate
│   ├── report.py            PDF report generation
│   └── nl_query.py          Natural language query agent (optional)
├── api/                     FastAPI backend and route handlers
├── config/                  Application configuration and settings
├── docs/                    Technical documentation
├── frontend/                Streamlit dashboard interface
├── orchestrator/            Pipeline sequencing and coordination
├── optional/                Optional feature modules (auth, exports)
├── tests/                   Test suite
├── utils/                   Shared utilities (LLM client, helpers)
├── Dockerfile               Container image definition
├── docker-compose.yml       Multi-service orchestration
├── requirements.txt         Python dependencies
├── .env.example             Environment variable template
└── .gitignore               Git exclusion rules
```

---

## Engineering Decisions

### Deterministic-First Approach

Every agent in the core pipeline produces identical outputs for identical inputs. This was a deliberate choice over a fully AI-driven architecture. Reproducibility is non-negotiable for production analytics. When a stakeholder asks why a number appears in a report, the answer must trace back to specific data and logic, not to a probabilistic model that may produce different output on the next run.

### Statistical Filtering

The validation engine enforces strict thresholds before any insight reaches the report. Correlations above 0.999 are suppressed (likely derived columns). Forecasts below 8 data points are dropped. Constant-value columns are excluded from KPI computation. These rules are not configurable at runtime. Quality gates should not be relaxable by users who may not understand the statistical implications.

### Executive-Grade PDF Layout

Reports are generated using ReportLab Platypus flowables, not manual canvas positioning. This ensures consistent pagination, proper text wrapping, and reliable rendering across document lengths. The narrative engine adapts commentary based on metric type rather than using generic phrasing.

### Modular Design

Each agent is independently testable, replaceable, and extensible. The orchestrator treats agents as interchangeable units with typed contracts. This design supports future scaling without requiring architectural changes.

---

## Limitations

This section documents known constraints honestly.

- The system processes data in-memory using Pandas. It is suitable for datasets under approximately one million rows. Larger datasets may require chunked processing, which is not yet implemented.
- There is no distributed execution layer. The pipeline runs on a single node.
- Forecasting supports univariate time-series only. Multivariate models (VAR, ARIMAX) are not yet available.
- The LLM-enhanced narrative layer depends on external API availability and adds latency to report generation.
- The system does not currently support real-time streaming data. It operates in batch mode.

---

## Future Roadmap

The following items represent planned technical enhancements:

- Multivariate forecasting support using VAR and ARIMAX models for cross-metric prediction.
- Chunked ingestion to handle datasets exceeding available memory.
- Persistent storage layer to replace file-based output management for multi-user deployments.
- Custom report templates allowing users to define layout and branding through configuration files.
- Distributed agent execution using task queues for horizontal scaling.
- Streaming data support for real-time analytical pipelines.

---

## Deployment

### Docker

Build and run both services with Docker Compose:

```bash
docker-compose up --build
```

This starts the FastAPI backend on port 8000 and the Streamlit frontend on port 8501. Upload and output directories are mounted as persistent volumes.

### Local

No additional infrastructure is required. The system runs entirely on localhost with file-based persistence.

---

This project is provided as-is for portfolio and demonstration purposes.
