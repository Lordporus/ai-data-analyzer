import os
import pickle
import logging
from pathlib import Path
from orchestrator.master import MasterOrchestrator

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


def is_redis_available(timeout: float = 2.0) -> bool:
    """
    Return True if Redis is reachable at REDIS_URL within *timeout* seconds.
    Handles both redis:// (plain) and rediss:// (TLS, e.g. Upstash) schemes.
    Used to decide whether to dispatch via Celery or fall back to sync execution.
    """
    try:
        import redis
        from urllib.parse import urlparse
        parsed = urlparse(REDIS_URL)
        host = parsed.hostname or "localhost"
        port = parsed.port or 6379
        db_num = int((parsed.path or "/0").lstrip("/") or 0)
        use_ssl = parsed.scheme == "rediss"
        r = redis.Redis(
            host=host,
            port=port,
            db=db_num,
            ssl=use_ssl,
            ssl_cert_reqs=None if use_ssl else "required",
            socket_connect_timeout=timeout,
            socket_timeout=timeout,
        )
        r.ping()
        return True
    except Exception:
        return False


def run_analysis_sync(
    file_path_str: str,
    output_dir_str: str,
    branding: dict = None,
    progress_callback=None,
    dataset_name: str = None,
) -> dict:
    """
    Synchronous fallback: runs the full pipeline in-process without Celery/Redis.
    Returns the same dict shape as the Celery task so callers need no changes.
    """
    file_path = Path(file_path_str)
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    orchestrator = MasterOrchestrator()

    try:
        result = orchestrator.run(
            csv_path=file_path,
            output_dir=output_dir,
            branding=branding,
            progress_callback=progress_callback,
            dataset_name=dataset_name,
        )

        # Persist the result as a pickle file (same convention as Celery path)
        pickle_path = output_dir / "pipeline_result.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(result, f)

        return {
            "status": result.status,
            "job_id": result.job_id,
            "pickle_path": str(pickle_path),
            "output_dir": str(output_dir),
            "summary": result.summary_dict() if hasattr(result, "summary_dict") else {},
            "errors": result.errors,
        }
    except Exception as e:
        logger.exception("Error executing synchronous pipeline")
        return {
            "status": "failed",
            "errors": [str(e)],
        }


# ── Celery (only initialised when Redis is available) ─────────────────────────
try:
    from celery import Celery

    celery_app = Celery(
        "analyzer",
        broker=REDIS_URL,
        backend=REDIS_URL,
    )

    celery_app.conf.update(
        task_track_started=True,
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
    )

    @celery_app.task(bind=True)
    def run_analysis_task(self, file_path_str: str, output_dir_str: str, branding: dict = None, dataset_name: str = None) -> dict:
        """
        Celery task to run the complete data analysis pipeline.
        Updates task state to show pipeline stage progress.
        """
        file_path = Path(file_path_str)
        output_dir = Path(output_dir_str)

        self.update_state(state="PROGRESS", meta={"stage": "Initializing pipeline...", "pct": 0.01})

        orchestrator = MasterOrchestrator()

        def celery_progress(stage: str, percentage: float):
            self.update_state(
                state="PROGRESS",
                meta={"stage": stage, "pct": percentage},
            )
            logger.info(f"Celery task progress: {stage} ({percentage * 100:.0f}%)")

        try:
            result = orchestrator.run(
                csv_path=file_path,
                output_dir=output_dir,
                branding=branding,
                progress_callback=celery_progress,
                dataset_name=dataset_name,
            )

            pickle_path = output_dir / "pipeline_result.pkl"
            with open(pickle_path, "wb") as f:
                pickle.dump(result, f)

            return {
                "status": result.status,
                "job_id": result.job_id,
                "pickle_path": str(pickle_path),
                "output_dir": str(output_dir),
                "summary": result.summary_dict() if hasattr(result, "summary_dict") else {},
                "errors": result.errors,
            }
        except Exception as e:
            logger.exception("Error executing Celery task")
            return {
                "status": "failed",
                "errors": [str(e)],
            }

except ImportError:
    # Celery not installed — sync-only mode is still fully functional
    logger.warning("Celery is not installed. Running in synchronous-only mode.")
    celery_app = None
    run_analysis_task = None
