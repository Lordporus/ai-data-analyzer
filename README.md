# AI Data Analyzer

**Transform Raw Data into Executive Intelligence in 15 Seconds.**

[View Live App](https://ai-data-analyst-59zk.onrender.com/)

---

## Overview

AI Data Analyzer is a professional SaaS platform that eliminates the friction of manual data analysis. It is a modular, multi-agent analytics pipeline that takes raw data (CSV, XLSX, SQL databases) and produces a complete analytical deliverable: cleaned data, statistical insights, time-series forecasts, an interactive dashboard, and a white-labeled, executive-grade PDF report.

Designed for business owners, consultants, and data analysts, the system handles every step from ingestion to final output deterministically and autonomously.

---

## Key Features

- **Multi-Format Data Ingestion**: Upload CSV and Excel files, or connect directly to live databases (PostgreSQL, BigQuery, Snowflake).
- **Automated Data Hygiene**: Smart handling of missing values, duplicates, mixed-type columns, and automatic schema inference.
- **Business-Aware Insights**: Generates actionable, plain-English executive summaries. Automatically identifies key trends, anomalies, and non-obvious correlations—completely free of corporate jargon and markdown symbols (markdown-free recommendations).
- **Auto Sector Detection**: Automatically detects business sector to frame insights.
- **Adaptive Forecasting**: Univariate and multivariate (VAR/ARIMAX) time-series forecasting. Smart forecast filtering automatically excludes identifiers like ID, age, and postal columns. 
- **Consulting-Grade PDF Reports**: Instantly generates branded PDF reports featuring custom logos, color schemes, and an executive summary (PDF report generation).
- **LLM AI Mode**: Integrated with Gemini 2.5 Flash for advanced strategic narratives.
- **Scheduled Deliveries**: Set up automated, recurring email reports using SendGrid.
- **Interactive Dashboard**: Explore your data through a dynamic Streamlit interface with real-time filters, live progress tracking, and session persistence.
- **Scalable Architecture**: Redis/Celery background processing for non-blocking concurrent user execution.
- **Team Workspaces**: Supabase auth with local fallback on network error, persistent analysis history, and team organization.

---

## Screenshots

*(Placeholder for Screenshots)*
- **Landing Page & Onboarding**
- **Interactive Dashboard & Charts**
- **Executive PDF Report Cover & Content**
- **Settings & Branding Panel**

---

## Tech Stack

The architecture is built for performance, modularity, and deterministic reliability:

- **Frontend**: Streamlit
- **Backend API**: FastAPI, Uvicorn
- **Task Queue & Async Execution**: Celery, Redis
- **Data Processing**: Pandas, NumPy, SciPy, Scikit-Learn, Statsmodels, OpenPyXL
- **Visualization**: Plotly
- **PDF Generation**: ReportLab (Platypus Flowables)
- **Database & Auth**: Supabase (PostgreSQL), Cloudflare R2
- **Email & Scheduling**: SendGrid, APScheduler
- **AI Integration (Optional)**: Google Gemini API (or any configured LLM via integration layer)

---

## Local Setup Instructions

### Prerequisites
- Python 3.11+
- Redis server (for task queuing)

### 1. Clone and Install

```bash
git clone https://github.com/Lordporus/ai-data-analyzer.git
cd ai-data-analyzer
pip install -r requirements.txt
```

### 2. Environment Configuration
Copy the template and fill in your variables:
```bash
cp .env.example .env
```

### 3. Start the Application
To run all services (Redis, Celery, FastAPI, Streamlit) simultaneously using the provided startup script:
```bash
bash start.sh
```

Alternatively, you can run them via Docker:
```bash
docker-compose up --build
```

Access the dashboard at `http://localhost:8501`.

---

## Environment Variables

For full functionality, configure the following in your `.env` file. Note that core deterministic features work without LLM keys.

```env
# Server
PORT=8501
API_HOST=0.0.0.0
API_PORT=8000

# LLM Configuration (Optional)
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash
LLM_API_KEY=your-api-key

# Database, Auth & Storage (Supabase & Cloudflare R2)
SUPABASE_URL=your-supabase-url
SUPABASE_ANON_KEY=your-anon-key
R2_ENDPOINT=your-r2-endpoint
R2_ACCESS_KEY=your-r2-access-key
R2_SECRET_KEY=your-r2-secret-key
R2_BUCKET=your-bucket-name
R2_PUBLIC_URL=your-r2-public-url

# Task Queue
REDIS_URL=redis://localhost:6379/0?ssl_cert_reqs=CERT_NONE
CELERY_BROKER_URL=redis://localhost:6379/0?ssl_cert_reqs=CERT_NONE

# Scheduled Reports
SENDGRID_API_KEY=your-sendgrid-api-key
SCHEDULER_ENABLED=true
```

---

## Render Deployment Instructions

The application is optimized for deployment on Render as a Dockerized Web Service.

1. **Prepare External Services**:
   - Provision a free **Supabase** project for Auth and Database.
   - Provision a free **Cloudflare R2** bucket for persistent file storage.
   - Provision an external **Redis** instance (e.g., Upstash) since Render's Free Tier does not include Redis. Update `REDIS_URL` accordingly.

2. **Deploy to Render**:
   - Create a new **Web Service** in the Render Dashboard.
   - Connect your GitHub repository.
   - Set the Environment to `Docker`.
   - Select the `Free` Instance Type.

3. **Configure Environment Variables**:
   In the Render dashboard, add the following variables:
   - `PYTHON_VERSION` = `3.11.0`
   - `PORT` = `8501`
   - `SUPABASE_URL` = `<your-supabase-url>`
   - `SUPABASE_ANON_KEY` = `<your-supabase-key>`
   - `REDIS_URL` = `<your-external-redis-url>?ssl_cert_reqs=CERT_NONE` *(Note: For Upstash Redis, append ?ssl_cert_reqs=CERT_NONE)*
   - `CELERY_BROKER_URL` = `<your-external-redis-url>?ssl_cert_reqs=CERT_NONE`
   - `LLM_PROVIDER` = `gemini`
   - `LLM_MODEL` = `gemini-2.5-flash`
   - `LLM_API_KEY` = `<your-gemini-key>`
   - `SCHEDULER_ENABLED` = `true`
   - `SENDGRID_API_KEY` = `<your-sendgrid-api-key>`

---

## Known Issues

- **Supabase Organizations Table Missing**: The `public.organizations` table may be missing in some Supabase instances. This is a non-critical issue as the system has an active fallback mechanism.

4. **Deploy**:
   - Click "Manual Deploy" -> "Deploy latest commit".
   - The provided `start.sh` script will automatically spin up the background workers and the Streamlit frontend within the single container.

---

**AI Data Analyzer** — Built for reliability, transparency, and actionable intelligence.
