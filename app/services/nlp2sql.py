import ast
import pandas as pd
from typing import Any

def clean_sql_query(sql_query: str) -> str:
    cleaned = sql_query.replace("SQLQuery:", "").strip()
    if cleaned.startswith("```sql"): cleaned = cleaned[6:]
    elif cleaned.startswith("```"):  cleaned = cleaned[3:]
    if cleaned.endswith("```"): cleaned = cleaned[:-3]
    return cleaned.strip()

def format_result(result: Any) -> Any:
    if not result:
        return [{"Result": "No results found"}]
    # Try to parse stringified python literals
    if isinstance(result, str):
        try:
            parsed = ast.literal_eval(result)
            result = parsed
        except Exception:
            return [{"Result": result}]
    if isinstance(result, list):
        if result and isinstance(result[0], tuple):
            # Convert list[tuple] to list[dict] with generic columns
            # or return raw tuples; here we return as list of lists
            return [list(row) for row in result]
        else:
            return result
    # Fallback
    return [str(result)]
