import logging
import uuid
from pathlib import Path
from apscheduler.schedulers.background import BackgroundScheduler
from config import settings
from orchestrator.master import MasterOrchestrator
from utils.emailer import send_report_email

logger = logging.getLogger(__name__)

# Setup scheduler
scheduler = BackgroundScheduler()

def run_and_email(email, dataset_path):
    """
    Job that runs the full analysis pipeline on the target file and emails the PDF.
    """
    logger.info("Starting scheduled analysis job for %s on file %s", email, dataset_path)
    try:
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            logger.error("Scheduled job failed: target dataset path %s does not exist", dataset_path)
            return

        # Unique run ID and output directory
        run_id = f"sched_{uuid.uuid4().hex[:8]}"
        output_dir = settings.OUTPUT_DIR / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run pipeline
        orchestrator = MasterOrchestrator()
        result = orchestrator.run(dataset_path, output_dir)

        if result.status == "completed" and result.pdf_report_path:
            logger.info("Analysis completed successfully for scheduled run %s. Sending email.", run_id)
            send_report_email(email, result.pdf_report_path, dataset_path.name)
        else:
            logger.error("Scheduled run %s failed to complete. Errors: %s", run_id, ", ".join(result.errors))
    except Exception as e:
        logger.exception("Error running scheduled pipeline job: %s", str(e))

def add_scheduled_job(job_id, day_of_week_val, email, dataset_path):
    """
    Add a scheduled job to the background scheduler.
    """
    if not settings.SCHEDULER_ENABLED:
        logger.warning("Scheduler is disabled in configuration. Job will not be added.")
        return False

    # Standard scheduled execution at 9:00 AM on the specified days
    # (or * for daily)
    scheduler.add_job(
        run_and_email,
        'cron',
        id=job_id,
        day_of_week=day_of_week_val,
        hour=9,
        minute=0,
        args=[email, dataset_path],
        replace_existing=True
    )
    
    if not scheduler.running:
        scheduler.start()
        logger.info("APScheduler started successfully.")
        
    logger.info("Added scheduled job %s for %s, repeating on: %s", job_id, email, day_of_week_val)
    return True
