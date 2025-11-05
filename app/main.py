import traceback
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app.models import PredictRequest, PredictResponse
from app.services.db import make_db_from_env
from app.services.llm import build_llm_and_prompt
from app.services.nlp2sql import clean_sql_query, format_result
from app.deps import load_env

# Load env ASAP
load_env()

app = FastAPI(title="NLP → SQL (FastAPI)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

# Globals initialized on startup
DB = None
LLM = None
QUERY_CHAIN = None

@app.on_event("startup")
def startup():
    global DB, LLM, QUERY_CHAIN
    DB = make_db_from_env()
    LLM, QUERY_CHAIN = build_llm_and_prompt(DB)

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question is required")
    try:
        sql_raw = QUERY_CHAIN.invoke({"question": q})
        sql = clean_sql_query(sql_raw)
        db_result = DB.run(sql)
        rows = format_result(db_result)

        final_prompt = f"""
Based on this SQL query: {sql}
And this result: {db_result}

Provide a clean, concise natural-language answer to the original question: {q}
Highlight key insights. Be brief.
"""
        final = LLM.invoke(final_prompt)
        explanation = final.content if hasattr(final, "content") else str(final)

        return PredictResponse(sql=sql, result=rows, explanation=explanation)
    except Exception as e:
        trace = traceback.format_exc()
        return PredictResponse(sql="", result=[], explanation="", error=f"{e}\n{trace}")

# ---------- Simple, nice UI (HTMX + Tailwind) ----------

@app.get("/", response_class=HTMLResponse)
def ui_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ui/predict", response_class=HTMLResponse)
def ui_predict(question: str = Form("")):
    if not question.strip():
        return '<div class="text-red-600">Please enter a question.</div>'
    try:
        sql_raw = QUERY_CHAIN.invoke({"question": question})
        sql = clean_sql_query(sql_raw)
        db_result = DB.run(sql)
        rows = format_result(db_result)

        final_prompt = f"""
Based on this SQL query: {sql}
And this result: {db_result}

Provide a clean, concise natural-language answer to the original question: {question}
Highlight key insights. Be brief.
"""
        final = LLM.invoke(final_prompt)
        explanation = final.content if hasattr(final, "content") else str(final)

        # Render a small card with results
        table_html = ""
        if isinstance(rows, list) and rows and isinstance(rows[0], list):
            # Render rows as simple table
            table_html += '<table class="w-full text-sm border mt-2">'
            for i, r in enumerate(rows):
                row_cls = "bg-gray-50" if i % 2 else ""
                table_html += f'<tr class="{row_cls}">' + "".join(
                    f'<td class="p-2 border">{str(c)}</td>' for c in r
                ) + "</tr>"
            table_html += "</table>"
        else:
            table_html = f'<pre class="text-xs mt-2">{rows}</pre>'

        return f"""
        <div class="p-4 bg-white rounded-2xl shadow">
            <div class="font-semibold mb-2">Generated SQL</div>
            <pre class="text-xs p-3 bg-gray-100 rounded">{sql}</pre>
            <div class="font-semibold mt-4 mb-2">Results</div>
            {table_html}
            <div class="font-semibold mt-4 mb-2">Explanation</div>
            <p class="text-sm leading-6">{explanation}</p>
        </div>
        """
    except Exception as e:
        return f'<div class="text-red-600">Error: {e}</div>'
