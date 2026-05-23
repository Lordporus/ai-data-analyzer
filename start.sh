#!/bin/bash
set -e

# Start Redis in the background
echo "Starting Redis..."
redis-server --daemonize yes

# Start Uvicorn API in the background
echo "Starting Uvicorn API..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Start Celery worker in the background
echo "Starting Celery Worker..."
celery -A utils.task_queue.celery_app worker --loglevel=info &

# Start Streamlit on the port provided by the environment (default 8501)
echo "Starting Streamlit..."
PORT=${PORT:-8501}
streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0
