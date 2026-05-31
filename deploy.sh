#!/usr/bin/env bash
# =============================================================================
# deploy.sh — Optimized production deploy for buildwithporus.qzz.io
# Run on EC2: ./deploy.sh
#
# Changes from previous approach:
#   ❌ Before: docker compose build --no-cache  (rebuilds ALL layers every time)
#   ✅ After:  docker compose build              (uses Docker layer cache)
#             Rolling restart per service       (no full downtime)
#
# Average deploy time: ~2-3 min (was 20+ min with --no-cache)
# =============================================================================
set -euo pipefail

APP_DIR="/home/ubuntu/ai-data-analyzer-main"
COMPOSE="docker compose"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀  AI Data Analyzer — Deploy Started"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── 1. Pull latest code ───────────────────────────────────────────────────────
echo "📥 Pulling latest code..."
cd "$APP_DIR"
git pull origin main

# ── 2. Build images using Docker cache ───────────────────────────────────────
# Docker layer cache means pip packages are only re-installed when
# requirements.txt changes. Code-only changes hit only the final COPY layer.
echo "🔨 Building Docker images (using cache)..."
$COMPOSE build

# ── 3. Rolling restart — one service at a time ──────────────────────────────
# --no-deps: don't restart dependent services unnecessarily
# --build:   use the image just built above (already cached, fast)
echo "🔄 Rolling restart: celery_worker..."
$COMPOSE up -d --no-deps celery_worker

echo "🔄 Rolling restart: api..."
$COMPOSE up -d --no-deps api

echo "🔄 Rolling restart: frontend..."
$COMPOSE up -d --no-deps frontend

# ── 4. Health check ──────────────────────────────────────────────────────────
echo "❤️  Waiting 5 seconds then checking health..."
sleep 5

if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health | grep -q "200"; then
    echo "✅ API health check passed"
else
    echo "⚠️  API health check failed — check logs: docker compose logs api"
fi

# ── 5. Show running containers ────────────────────────────────────────────────
echo ""
echo "📦 Running containers:"
$COMPOSE ps

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅  Deploy Complete — https://buildwithporus.qzz.io"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
