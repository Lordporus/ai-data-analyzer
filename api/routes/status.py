from fastapi import APIRouter, HTTPException
from utils.task_queue import run_analysis_task
from pathlib import Path

router = APIRouter()

@router.get("/status/{job_id}")
def get_job_status(job_id: str):
    """
    Get the real-time execution status of an analysis job in the task queue.
    """
    if run_analysis_task is None:
        raise HTTPException(
            503,
            "Task queue (Celery) is not available — Redis is unreachable or not configured."
        )
    task = run_analysis_task.AsyncResult(job_id)
    
    response = {
        "job_id": job_id,
        "status": task.state, # PENDING, STARTED, PROGRESS, SUCCESS, FAILURE, RETRY
        "info": None
    }
    
    if task.state == "PROGRESS":
        # Progress metadata is stored in task.info (meta dict)
        response["status"] = "running"
        response["info"] = task.info
    elif task.state == "SUCCESS":
        result = task.result
        if not result:
            response["status"] = "failed"
            response["info"] = {"errors": ["No result returned from task."]}
            return response
            
        status = result.get("status", "failed")
        if status in ("completed", "completed_with_warnings"):
            response["status"] = "completed"
            
            output_dir_str = result.get("output_dir", "")
            output_id = Path(output_dir_str).name
            base = f"/outputs/{output_id}"
            
            response["info"] = {
                "status": status,
                "summary": result.get("summary", {}),
                "errors": result.get("errors", []),
                "downloads": {
                    "cleaned_csv": f"{base}/cleaned_data.csv",
                    "dashboard_html": f"{base}/dashboard.html",
                    "pdf_report": f"{base}/report.pdf",
                    "markdown_report": f"{base}/report.md",
                    "excel_report": f"{base}/analysis_report.xlsx",
                }
            }
        else:
            response["status"] = "failed"
            response["info"] = {
                "status": status,
                "errors": result.get("errors", ["Analysis pipeline execution failed."])
            }
    elif task.state == "FAILURE":
        response["status"] = "failed"
        response["info"] = {
            "errors": [str(task.result) if task.result else "Task failed with exception."]
        }
    else:
        # If task state is PENDING or RECEIVED, report as queued
        response["status"] = "queued"
        response["info"] = {"stage": "Queue waiting...", "pct": 0.0}
        
    return response
