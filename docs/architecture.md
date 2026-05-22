# System Architecture Overview

---

## High-Level Overview

The Adaptive Agentic Data Intelligence System is a modular analytics pipeline composed of specialized, single-responsibility agents. Each agent performs a scoped operation on structured input and passes a typed result to the next stage. The system is designed around three principles:

1. **Modular agent architecture** -- Each processing stage is encapsulated in an independent agent with a defined input/output contract. Agents can be tested, replaced, or extended in isolation.
2. **Deterministic-first design** -- Every core computation produces identical outputs for identical inputs. There are no stochastic steps in the critical path.
3. **Optional AI enhancement** -- LLM-powered features (natural language querying, narrative augmentation) are layered on top of the deterministic core. If unavailable, the system produces the same validated output.

---

## Execution Flow

The pipeline executes in strict sequential order. No agent bypasses or reorders are permitted at runtime.

```
CSV Upload
    |
    v
IngestionAgent -----> Schema detection, type inference, metadata extraction
    |
    v
CleaningAgent ------> Null imputation, duplicate removal, type coercion
    |
    v
RepairAgent --------> Structural normalization, column renaming, encoding cleanup
    |
    v
DataQualityAgent ---> Quality scoring (completeness, consistency, validity)
    |
    v
InsightAgent -------> KPIs, trend analysis, correlation matrix, risk/opportunity detection
    |
    v
ForecastAgent ------> Adaptive time-series forecasting (Linear / Holt-Winters)
    |
    v
ReportValidationEngine --> Statistical filtering, NaN sanitization, derived column suppression
    |
    v
ReportAgent --------> Executive PDF report generation (ReportLab Platypus)
```

Each transition passes a typed dataclass. The orchestrator manages sequencing and aggregates results into a single `PipelineResult` object consumed by the frontend and API layers.

---

## Component Responsibilities

### Orchestrator

**Location**: `orchestrator/master.py`

Coordinates the full pipeline execution. Receives a file path, invokes each agent in sequence, collects results into a `PipelineResult` dataclass, and handles top-level error recovery. The orchestrator does not perform any analytical logic itself.

### Agents

**Location**: `agents/`

Each agent inherits from `BaseAgent` and implements a `run()` method. Agents are stateless: they receive input, produce output, and retain no session information. The agent roster:

| Agent | Responsibility |
|:------|:---------------|
| `IngestionAgent` | CSV parsing, column type detection, metadata extraction |
| `CleaningAgent` | Null handling, duplicate removal, type coercion |
| `RepairAgent` | Structural repair, column normalization |
| `DataQualityAgent` | Quality scoring across completeness, consistency, validity |
| `InsightAgent` | Statistical analysis, KPI computation, trend/correlation detection |
| `ForecastAgent` | Time-series forecasting with adaptive model selection |
| `NLQueryAgent` | Natural language data querying (optional, requires LLM) |

### Validation Engine

**Location**: `agents/report_validation.py`

Acts as a quality gate between raw analysis and the presentation layer. Filters statistically weak results, suppresses derived columns, sanitizes invalid values, and enforces minimum thresholds. The validation engine prevents misleading or trivial insights from reaching the final report.

### Report Layer

**Location**: `agents/report.py`

Generates executive-grade PDF reports using ReportLab Platypus flowables. The report layer consumes validated `InsightResult` and `ForecastResult` objects. It applies a metric-aware narrative engine that adapts commentary based on metric type (volume, efficiency, dispersion). All raw statistical artifacts are stripped from the output.

### Frontend Interface

**Location**: `frontend/`

Streamlit-based interactive dashboard. Provides filters, KPI snapshots, smart charting with auto-defaults, chart overlays (trendline, rolling average, outlier highlighting), drill-down analytics with breadcrumb navigation, and export utilities. The frontend consumes the same `PipelineResult` object as the report layer.

---

## Design Philosophy

### Deterministic Reliability First

Every agent in the core pipeline produces identical outputs for identical inputs. This guarantees:

- **Reproducibility** -- The same CSV always produces the same report.
- **Testability** -- Unit tests can assert exact outputs without tolerance ranges.
- **Auditability** -- Results can be traced back to specific data and logic paths.

### AI as Augmentation, Not Dependency

LLM integration is confined to two optional features: natural language querying (`NLQueryAgent`) and narrative enhancement in the report layer. Neither is in the critical path. If the LLM is unreachable, times out, or returns an error, the system falls back to template-based output with zero functional degradation.

### Guardrail-Based Reporting

The `ReportValidationEngine` enforces strict statistical thresholds before any insight reaches the report. This prevents:

- Trivial correlations from appearing as discoveries.
- Forecasts based on insufficient data from being presented as projections.
- Constant-value columns from being reported as KPIs.
- Derived or structural columns from polluting analytical sections.

---

## Fault Tolerance

### Forecast Fallback Logic

If Holt-Winters exponential smoothing fails (insufficient seasonal periods, convergence failure), the engine automatically falls back to linear regression. If linear regression also fails (fewer than 8 data points), the forecast is suppressed entirely.

### Missing Data Handling

- **Numeric columns**: Imputed with column median during the cleaning phase.
- **Categorical columns**: Imputed with column mode.
- **Datetime columns**: Rows with unparseable dates are dropped; the system logs the count of dropped rows.
- **Entire columns**: Columns with >90% null values are flagged in the quality report but retained for transparency.

### NaN Sanitization

The validation engine scans all forecast arrays for NaN and Inf values. Any forecast containing invalid values is dropped before reaching the report layer. This prevents rendering failures in the PDF generation step.

### Statistical Suppression Rules

| Condition | Action |
|:----------|:-------|
| Fewer than 8 data points | Forecast suppressed |
| R-squared below 0.1 | Trend suppressed |
| Correlation above 0.999 | Pair suppressed (likely derived) |
| Column variance below 1e-9 | KPI suppressed (constant value) |
| Column matches structural pattern | Excluded from all analysis |

These rules are applied deterministically and are not configurable at runtime to prevent accidental relaxation of quality standards.
