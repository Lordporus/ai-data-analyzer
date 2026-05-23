FROM python:3.11-slim

WORKDIR /app

# Install dependencies and Redis
COPY requirements.txt .
RUN apt-get update && \
    apt-get install -y redis-server && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure start.sh is executable
RUN chmod +x start.sh

# Create runtime directories
RUN mkdir -p uploads outputs

# Expose ports
EXPOSE 8000 8501

# Default: run the combined start script
CMD ["bash", "start.sh"]
