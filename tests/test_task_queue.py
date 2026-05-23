import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.task_queue import run_analysis_task
from api.routes.status import get_job_status
from orchestrator.master import PipelineResult

class TestTaskQueueSystem(unittest.TestCase):
    @patch("utils.task_queue.MasterOrchestrator")
    @patch("utils.task_queue.pickle")
    @patch("utils.task_queue.open")
    def test_run_analysis_task_success(self, mock_open, mock_pickle, mock_orchestrator_class):
        mock_orchestrator = MagicMock()
        mock_result = MagicMock(spec=PipelineResult)
        mock_result.status = "completed"
        mock_result.job_id = "test_job_123"
        mock_result.errors = []
        mock_result.summary_dict.return_value = {"row_count": 100}
        
        mock_orchestrator.run.return_value = mock_result
        mock_orchestrator_class.return_value = mock_orchestrator
        
        # Since bind=True automatically injects the task instance as self,
        # we mock 'update_state' directly on the task instance.
        run_analysis_task.update_state = MagicMock()
        
        # Call .run() directly to bypass Celery's task wrapping decorator without manual self
        result_dict = run_analysis_task.run("dummy.csv", "out_dir", None)
        
        mock_orchestrator.run.assert_called_once()
        run_analysis_task.update_state.assert_called()
        self.assertEqual(result_dict["status"], "completed")
        self.assertEqual(result_dict["job_id"], "test_job_123")
        self.assertEqual(result_dict["summary"]["row_count"], 100)
        
    @patch("api.routes.status.run_analysis_task.AsyncResult")
    def test_get_job_status_progress(self, mock_async_result_class):
        mock_async_result = MagicMock()
        mock_async_result.state = "PROGRESS"
        mock_async_result.info = {"stage": "Cleaning...", "pct": 0.3}
        mock_async_result_class.return_value = mock_async_result
        
        response = get_job_status("job_abc")
        
        mock_async_result_class.assert_called_once_with("job_abc")
        self.assertEqual(response["job_id"], "job_abc")
        self.assertEqual(response["status"], "running")
        self.assertEqual(response["info"]["stage"], "Cleaning...")
        self.assertEqual(response["info"]["pct"], 0.3)

    @patch("api.routes.status.run_analysis_task.AsyncResult")
    def test_get_job_status_success(self, mock_async_result_class):
        mock_async_result = MagicMock()
        mock_async_result.state = "SUCCESS"
        mock_async_result.result = {
            "status": "completed",
            "job_id": "job_abc",
            "output_dir": "/path/to/outputs/job_abc",
            "summary": {"row_count": 50},
            "errors": []
        }
        mock_async_result_class.return_value = mock_async_result
        
        response = get_job_status("job_abc")
        
        mock_async_result_class.assert_called_once_with("job_abc")
        self.assertEqual(response["job_id"], "job_abc")
        self.assertEqual(response["status"], "completed")
        self.assertEqual(response["info"]["downloads"]["cleaned_csv"], "/outputs/job_abc/cleaned_data.csv")

if __name__ == "__main__":
    unittest.main()
