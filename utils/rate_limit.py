import os
import time
import logging
from fastapi import Request, HTTPException

logger = logging.getLogger(__name__)

_memory_store = {}

def rate_limit_dependency(request: Request):
    """
    Dependency to enforce rate limits per IP address (60 req/min).
    Uses Redis if available, else falls back to in-memory store.
    """
    client_ip = request.client.host if request.client else "unknown"
    limit = 60
    window = 60
    current_time = int(time.time())
    
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        try:
            import redis
            from urllib.parse import urlparse
            parsed = urlparse(redis_url)
            host = parsed.hostname or "localhost"
            port = parsed.port or 6379
            db_num = int((parsed.path or "/0").lstrip("/") or 0)
            use_ssl = parsed.scheme == "rediss"
            r = redis.Redis(host=host, port=port, db=db_num, ssl=use_ssl, ssl_cert_reqs=None if use_ssl else "required", socket_timeout=1.0)
            
            key = f"rate_limit:{client_ip}"
            requests = r.get(key)
            
            if requests and int(requests) >= limit:
                raise HTTPException(status_code=429, detail="Too Many Requests")
            
            p = r.pipeline()
            p.incr(key)
            p.expire(key, window)
            p.execute()
            return
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"Redis rate limit failed, falling back to memory: {e}")
            pass
            
    key = f"rate_limit:{client_ip}"
    if key not in _memory_store:
        _memory_store[key] = {"count": 1, "expires": current_time + window}
    else:
        if current_time > _memory_store[key]["expires"]:
            _memory_store[key] = {"count": 1, "expires": current_time + window}
        else:
            if _memory_store[key]["count"] >= limit:
                raise HTTPException(status_code=429, detail="Too Many Requests")
            _memory_store[key]["count"] += 1
