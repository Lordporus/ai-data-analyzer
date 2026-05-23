import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.stream_connector import poll_once, poll_api_stream
from api.main import app
from fastapi.testclient import TestClient

class TestStreamConnector(unittest.TestCase):
    @patch("utils.stream_connector.requests.get")
    def test_poll_once_success(self, mock_get):
        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"Date": "2026-05-23 00:00:00", "Sales": 150.0, "Quantity": 5},
            {"Date": "2026-05-23 00:30:00", "Sales": 180.5, "Quantity": 3}
        ]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        df = poll_once("http://mock-url.com/api/data")
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertIn("Sales", df.columns)
        self.assertEqual(df["Sales"].iloc[0], 150.0)
        mock_get.assert_called_once_with("http://mock-url.com/api/data", timeout=10)

    @patch("utils.stream_connector.poll_once")
    @patch("utils.stream_connector.time.sleep", side_effect=InterruptedError("Stop loop"))
    def test_poll_api_stream_callback(self, mock_sleep, mock_poll_once):
        # Mock returned data
        mock_df = pd.DataFrame([{"Sales": 100}])
        mock_poll_once.return_value = mock_df
        
        callback_mock = MagicMock()
        
        # Expect InterruptedError to break the infinite loop
        with self.assertRaises(InterruptedError):
            poll_api_stream("http://mock-url.com/api/data", interval_seconds=10, callback=callback_mock)
            
        callback_mock.assert_called_once_with(mock_df)

    def test_fastapi_mock_stream_route(self):
        client = TestClient(app)
        response = client.get("/api/mock-stream")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        
        # Verify schema elements
        first_item = data[0]
        self.assertIn("Date", first_item)
        self.assertIn("Sales", first_item)
        self.assertIn("Quantity", first_item)
        self.assertIn("Store_ID", first_item)
        self.assertIn("Category", first_item)

if __name__ == "__main__":
    unittest.main()
