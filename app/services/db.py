import os
from urllib.parse import quote_plus
from langchain_community.utilities import SQLDatabase

def make_db_from_env() -> SQLDatabase:
    username = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    host     = os.getenv("DB_HOST")
    port     = os.getenv("DB_PORT", "3306")
    database = os.getenv("DB_NAME")

    if not all([username, password, host, database]):
        missing = [k for k,v in {
            "DB_USER": username, "DB_PASSWORD": password,
            "DB_HOST": host, "DB_NAME": database
        }.items() if not v]
        raise ValueError(f"Missing DB env vars: {', '.join(missing)}")

    password_encoded = quote_plus(password)
    mysql_uri = f"mysql+pymysql://{username}:{password_encoded}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(mysql_uri)
