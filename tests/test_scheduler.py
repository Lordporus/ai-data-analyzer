import sys
import unittest
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings
from utils.scheduler import add_scheduled_job, scheduler

class TestSchedulerSystem(unittest.TestCase):
    def setUp(self):
        # Temporarily enable scheduler for test
        self.original_enabled = settings.SCHEDULER_ENABLED
        settings.SCHEDULER_ENABLED = True

    def tearDown(self):
        settings.SCHEDULER_ENABLED = self.original_enabled
        # Clean up added test jobs
        for job in list(scheduler.get_jobs()):
            if job.id.startswith("test_"):
                job.remove()

    def test_add_scheduled_job(self):
        dummy_file = ROOT / "tests" / "sample_data.csv"
        job_id = "test_sched_run"
        email = "test@company.com"
        
        success = add_scheduled_job(
            job_id=job_id,
            day_of_week_val="mon",
            email=email,
            dataset_path=str(dummy_file)
        )
        
        self.assertTrue(success)
        
        # Verify job is scheduled in APScheduler
        job = scheduler.get_job(job_id)
        self.assertIsNotNone(job)
        self.assertEqual(job.args[0], email)
        self.assertEqual(job.args[1], str(dummy_file))
        
        # Verify it has correct trigger/recurrence configuration
        self.assertEqual(str(job.trigger.fields[4]), "mon") # day_of_week
        self.assertEqual(str(job.trigger.fields[5]), "9") # hour
        self.assertEqual(str(job.trigger.fields[6]), "0") # minute
