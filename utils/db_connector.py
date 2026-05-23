import logging
import pandas as pd
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)

def connect_postgres(host, port, db, user, password, query):
    """
    Connect to a PostgreSQL database and fetch query results as a DataFrame.
    """
    logger.info("Connecting to PostgreSQL at %s:%s/%s", host, port, db)
    connection_url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    engine = create_engine(connection_url)
    try:
        df = pd.read_sql(query, engine)
        logger.info("Successfully fetched %d rows from PostgreSQL", len(df))
        return df
    finally:
        engine.dispose()

def connect_bigquery(project_id, query, credentials_json_path=None):
    """
    Connect to Google BigQuery and fetch query results as a DataFrame.
    credentials_json_path can be a path to a service account JSON file, or None.
    """
    logger.info("Connecting to Google BigQuery, project: %s", project_id)
    from google.cloud import bigquery
    
    if credentials_json_path:
        client = bigquery.Client.from_service_account_json(credentials_json_path, project=project_id)
    else:
        client = bigquery.Client(project=project_id)
        
    query_job = client.query(query)
    df = query_job.to_dataframe()
    logger.info("Successfully fetched %d rows from BigQuery", len(df))
    return df

def connect_snowflake(account, user, password, database, schema, warehouse, query):
    """
    Connect to Snowflake and fetch query results as a DataFrame.
    """
    logger.info("Connecting to Snowflake database: %s", database)
    import snowflake.connector
    
    ctx = snowflake.connector.connect(
        user=user,
        password=password,
        account=account,
        warehouse=warehouse,
        database=database,
        schema=schema
    )
    try:
        cs = ctx.cursor()
        try:
            cs.execute(query)
            df = cs.fetch_pandas_all()
            logger.info("Successfully fetched %d rows from Snowflake", len(df))
            return df
        finally:
            cs.close()
    finally:
        ctx.close()
