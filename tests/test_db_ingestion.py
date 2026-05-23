import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.ingestion import IngestionAgent, IngestionResult
from utils.db_connector import connect_postgres, connect_bigquery, connect_snowflake

class TestDatabaseIngestion(unittest.TestCase):
    def test_direct_dataframe_ingestion(self):
        # Create a sample DataFrame with 50 rows so categorical detection triggers
        data = {
            "Date": [f"2026-01-{i:02d}" for i in range(1, 51)],
            "Sales": [100.0 + i for i in range(50)],
            "Category": ["A" if i % 2 == 0 else "B" for i in range(50)]
        }
        df = pd.DataFrame(data)
        
        agent = IngestionAgent()
        result = agent.run(df)
        
        self.assertIsInstance(result, IngestionResult)
        self.assertEqual(result.row_count, 50)
        self.assertEqual(result.col_count, 3)
        self.assertIn("Date", result.column_names)
        self.assertEqual(result.detected_types["Sales"], "numeric")
        self.assertEqual(result.detected_types["Category"], "categorical")
        self.assertEqual(result.file_size_bytes, 0)

    @patch("utils.db_connector.create_engine")
    @patch("utils.db_connector.pd.read_sql")
    def test_connect_postgres(self, mock_read_sql, mock_create_engine):
        mock_df = pd.DataFrame({"col1": [1, 2]})
        mock_read_sql.return_value = mock_df
        
        df = connect_postgres(
            host="localhost",
            port=5432,
            db="test_db",
            user="user",
            password="pwd",
            query="SELECT * FROM table"
        )
        
        mock_create_engine.assert_called_once_with("postgresql://user:pwd@localhost:5432/test_db")
        mock_read_sql.assert_called_once()
        self.assertEqual(len(df), 2)

    @patch("google.cloud.bigquery.Client")
    def test_connect_bigquery(self, mock_client_class):
        mock_client = MagicMock()
        mock_query_job = MagicMock()
        mock_df = pd.DataFrame({"col1": [3, 4]})
        
        mock_client.query.return_value = mock_query_job
        mock_query_job.to_dataframe.return_value = mock_df
        mock_client_class.return_value = mock_client
        
        df = connect_bigquery(
            project_id="test-project",
            query="SELECT * FROM table",
            credentials_json_path=None
        )
        
        mock_client_class.assert_called_once_with(project="test-project")
        mock_client.query.assert_called_once_with("SELECT * FROM table")
        self.assertEqual(len(df), 2)

    def test_connect_snowflake(self):
        mock_snowflake = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_df = pd.DataFrame({"col1": [5, 6]})
        
        mock_snowflake.connector.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetch_pandas_all.return_value = mock_df
        
        with patch.dict("sys.modules", {"snowflake": mock_snowflake, "snowflake.connector": mock_snowflake.connector}):
            df = connect_snowflake(
                account="sf-acc",
                user="sf-user",
                password="sf-pass",
                database="sf-db",
                schema="sf-schema",
                warehouse="sf-wh",
                query="SELECT * FROM table"
            )
            
            mock_snowflake.connector.connect.assert_called_once_with(
                user="sf-user",
                password="sf-pass",
                account="sf-acc",
                warehouse="sf-wh",
                database="sf-db",
                schema="sf-schema"
            )
            mock_cursor.execute.assert_called_once_with("SELECT * FROM table")
            self.assertEqual(len(df), 2)

if __name__ == "__main__":
    unittest.main()
