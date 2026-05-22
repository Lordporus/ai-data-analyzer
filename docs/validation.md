# Report Validation and Guardrail System

---

## Purpose of the Validation Layer

The `ReportValidationEngine` is the final quality gate between raw analytical output and the presentation layer. Its purpose is to ensure that every insight, trend, correlation, and forecast that appears in the executive report or interactive dashboard meets minimum standards of statistical validity and business relevance.

Without this layer, the system would surface noise as signal: trivial correlations between derived columns, forecasts built on three data points, KPIs computed from constant-value columns, and trend lines with negligible explanatory power. The validation engine prevents these artifacts from reaching decision-makers.

**Location**: `agents/report_validation.py`

---

## Derived Column Detection

### Problem

Datasets frequently contain structural columns that have no analytical value: row identifiers, auto-increment keys, GUIDs, unnamed index columns, and internal tracking fields. If these columns enter the analysis pipeline, they produce meaningless KPIs (e.g., "Average ID: 5042") and spurious correlations (e.g., "ID is highly correlated with Row Number").

### Implementation

The validation engine matches column names against a set of structural patterns:

| Pattern | Examples |
|:--------|:---------|
| `^id$` | id |
| `^index$` | index |
| `^row_?id$` | row_id, rowid |
| `^guid$`, `^uuid$` | guid, uuid |
| `^key$` | key |
| `^_` | _internal, _temp |
| `unnamed` | Unnamed: 0, unnamed_col |
| `auto_increment` | auto_increment |

Matching is case-insensitive and uses regex. Any column matching these patterns is excluded from KPI computation, correlation analysis, trend detection, and forecast generation.

---

## Correlation Filtering

### Problem

Correlation matrices frequently contain entries that are mathematically valid but analytically useless:

1. **Self-correlations** (r = 1.0) -- A column is always perfectly correlated with itself.
2. **Derived pairs** (r > 0.999) -- Columns like `price` and `price_with_tax` are algebraically related, not statistically.
3. **Near-zero correlations** -- Weak relationships that do not warrant executive attention.

### Implementation

The validation engine applies the following filters to the correlation matrix:

| Filter | Threshold | Action |
|:-------|:----------|:-------|
| Derived pair detection | r > 0.999 | Pair excluded from report |
| Structural column involvement | Any column matches structural pattern | Pair excluded |
| Weak correlation | Below minimum reporting threshold | Not surfaced in highlights |

After filtering, only correlations that represent genuine, non-obvious relationships between business-relevant columns are included in the report.

---

## NaN and Inf Sanitization

### Problem

NaN (Not a Number) and Inf (Infinity) values can appear in analytical outputs when:

- Division by zero occurs during percentage calculations.
- A forecast model fails to converge on certain data points.
- Aggregation functions operate on entirely null columns.
- Rolling statistics are computed on windows shorter than the minimum period.

If these values reach the report layer, they cause rendering failures (ReportLab cannot typeset `NaN`), misleading visualizations, and broken chart axes.

### Implementation

The validation engine performs value-level scanning on all numerical outputs:

- **Forecast arrays**: Any forecast containing one or more NaN values is dropped entirely.
- **KPI values**: KPIs with NaN or Inf values are excluded from the report.
- **Trend statistics**: Trends with NaN slope, R-squared, or p-value are suppressed.

Sanitization occurs after computation and before serialization, ensuring that the report and dashboard layers never encounter invalid numerical values.

---

## Forecast Suppression Rules

### Problem

Time-series forecasts require minimum data density to produce statistically meaningful results. A forecast built on 4 data points has no predictive validity but may appear visually convincing in a chart. Presenting such forecasts as projections is misleading.

### Implementation

The validation engine enforces the following suppression rules on each forecast metric:

| Rule | Threshold | Effect |
|:-----|:----------|:-------|
| Minimum historical length | 8 data points | Forecast dropped |
| NaN in forecast values | Any NaN present | Forecast dropped |
| Structural column | Column matches derived pattern | Forecast dropped |

If all forecasts for a dataset are suppressed, the validation engine returns `None`. The report and dashboard layers handle this gracefully by omitting the forecast section entirely.

---

## Prevention of Statistical Hallucination

### Problem

Statistical hallucination occurs when a system presents computed artifacts as meaningful insights. Common examples:

- Reporting a "trend" on a column with zero variance (a flat line always has R-squared = undefined or 0).
- Highlighting a "correlation" between an ID column and a sequential metric (both increase monotonically).
- Generating a "forecast" from 3 data points and presenting it with confidence intervals.
- Labeling a column mean as a "KPI" when the column represents a technical identifier.

These outputs are technically computable but analytically meaningless. Presenting them undermines trust in the system.

### Implementation

The validation engine addresses each category:

| Hallucination Type | Detection | Prevention |
|:-------------------|:----------|:-----------|
| Flat-line trends | Slope below 0.0001, variance below 1e-9 | Trend suppressed |
| Derived correlations | r > 0.999, structural column detected | Pair excluded |
| Insufficient forecasts | Fewer than 8 data points | Forecast not generated |
| Meaningless KPIs | Column matches structural pattern, zero variance | KPI excluded |

---

## Executive Integrity Enforcement

The validation layer enforces a principle: every element in the final report must be defensible. If a stakeholder asks "why does this appear in the report?", the answer must reference a genuine statistical finding, not a computational artifact.

### What Passes Validation

- KPIs derived from business-relevant columns with measurable variance.
- Trends with R-squared above the minimum threshold (0.1), indicating at least weak explanatory power.
- Correlations between non-derived columns with absolute correlation below 0.999 and above the minimum reporting threshold.
- Forecasts built on 8+ data points with no NaN values in the projection array.

### What Does Not Pass

- Any output involving structural or derived columns.
- Forecasts with fewer than 8 data points.
- Correlations that are algebraically inevitable rather than statistically discovered.
- KPIs with zero variance.
- Any numerical output containing NaN or Inf.

The validation engine is not configurable at runtime. Thresholds are set in the class constructor and are not exposed as user-facing parameters. This is a deliberate design choice: quality gates should not be relaxable by end users who may not understand the statistical implications of threshold changes.
