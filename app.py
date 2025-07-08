import os
import asyncio
import logging
from io import BytesIO

import pandas as pd
import cohere
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Path, Request
from sqlalchemy import create_engine, text
from starlette.responses import StreamingResponse, JSONResponse

# ─── Load .env & Configure Cohere ──────────────────────────────────────────────
load_dotenv()  # reads .env from project root
COHERE_KEY = os.getenv("COHERE_API_KEY")
MODEL_ID = os.getenv("COHERE_MODEL_ID")  # e.g. "command-medium-nightly"
if not COHERE_KEY:
    raise RuntimeError("COHERE_API_KEY not found—please set it in your .env")
if not MODEL_ID:
    raise RuntimeError("COHERE_MODEL_ID not found—please set it in your .env")
co = cohere.Client(COHERE_KEY)

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kkc_api")
logger.info("Using Cohere model: %s", MODEL_ID)

# ─── Database Setup ───────────────────────────────────────────────────────────
DB_USER = "root"
DB_PASSWORD = "1234"
DB_HOST = "localhost"
DB_PORT = "3306"
DB_NAME = "kkc"
DATABASE_URL = (
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)
engine = create_engine(DATABASE_URL, echo=True)

# ─── FastAPI Init ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="KKC Database API",
    description="List tables, export to Excel, extended dashboard & Cohere insights per table",
    version="1.0.0"
)

# ─── Global Exception Handler ─────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}  # return actual error for dashboard troubleshooting
    )

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_tables() -> list[str]:
    with engine.connect() as conn:
        result = conn.execute(text("SHOW TABLES;"))
        return [r[0] for r in result]

async def _cohere_generate(prompt: str):
    """
    Offload the blocking Cohere generate() to a worker thread.
    """
    return await asyncio.to_thread(
        lambda: co.generate(
            model=MODEL_ID,
            prompt=prompt,
            max_tokens=100,
            temperature=0.5,
            stop_sequences=["--"]
        )
    )

# numeric types for stats
NUMERIC_TYPES = ("int", "bigint", "smallint", "decimal", "float", "double", "numeric")

# ─── Static Endpoints ──────────────────────────────────────────────────────────

@app.get("/tables", summary="List all tables")
def list_tables():
    return {"tables": _get_tables()}

@app.get("/health", summary="Health check")
def health_check():
    return {"status": "ok"}

@app.get(
    "/{table_name}/dashboard",
    summary="Extended dashboard: schema, quality, stats, sample"
)
def dashboard(
    table_name: str = Path(..., description="Name of the table")
):
    try:
        tables = _get_tables()
        if table_name not in tables:
            raise HTTPException(404, f"Table '{table_name}' not found")

        # fetch counts, columns, sample
        with engine.connect() as conn:
            row_count = conn.execute(text(f"SELECT COUNT(*) FROM `{table_name}`")).scalar()
            columns = [r[0] for r in conn.execute(text(f"SHOW COLUMNS FROM `{table_name}`"))]
            sample_df = pd.read_sql_query(text(f"SELECT * FROM `{table_name}` LIMIT 5"), conn)

            quality = {}
            stats = {}
            for col in columns:
                # get data type
                stmt = text(
                    "SELECT DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS "
                    "WHERE TABLE_SCHEMA = :schema AND TABLE_NAME = :table AND COLUMN_NAME = :col"
                )
                dtype = conn.execute(stmt, {"schema": DB_NAME, "table": table_name, "col": col}).scalar()

                # null & distinct counts
                null_count = conn.execute(text(
                    f"SELECT COUNT(*) - COUNT(`{col}`) FROM `{table_name}`"
                )).scalar()
                distinct_count = conn.execute(text(
                    f"SELECT COUNT(DISTINCT `{col}`) FROM `{table_name}`"
                )).scalar()
                quality[col] = {
                    "dtype": dtype,
                    "null_count": int(null_count),
                    "distinct_count": int(distinct_count)
                }

                # descriptive stats based on dtype
                if dtype in NUMERIC_TYPES:
                    r = conn.execute(text(
                        f"SELECT MIN(`{col}`), MAX(`{col}`), AVG(`{col}`), STDDEV_POP(`{col}`) FROM `{table_name}`"
                    )).fetchone()
                    stats[col] = {
                        "min": r[0],
                        "max": r[1],
                        "avg": float(r[2]) if r[2] is not None else None,
                        "std": float(r[3]) if r[3] is not None else None
                    }
                elif dtype in ("datetime", "date", "timestamp"):  # time cols
                    r = conn.execute(text(
                        f"SELECT MIN(`{col}`), MAX(`{col}`) FROM `{table_name}`"
                    )).fetchone()
                    stats[col] = {"earliest": str(r[0]), "latest": str(r[1])}
                else:
                    # top 3 frequencies
                    freq = conn.execute(text(
                        f"SELECT `{col}`, COUNT(*) as cnt FROM `{table_name}` "
                        f"GROUP BY `{col}` ORDER BY cnt DESC LIMIT 3"
                    )).fetchall()
                    stats[col] = [{"value": f[0], "count": f[1]} for f in freq]

        return {
            "table": table_name,
            "metrics": {"row_count": row_count, "column_count": len(columns)},
            "schema": columns,
            "quality": quality,
            "stats": stats,
            "sample_rows": sample_df.to_dict(orient="records")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in dashboard for %s: %s", table_name, e)
        raise HTTPException(500, detail=str(e))

@app.get(
    "/{table_name}/excel",
    summary="Download entire table as Excel"
)
def download_excel(
    table_name: str = Path(..., description="Name of the table")
):
    tables = _get_tables()
    if table_name not in tables:
        raise HTTPException(404, f"Table '{table_name}' not found")

    df = pd.read_sql_query(text(f"SELECT * FROM `{table_name}`"), engine)
    out = BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as w:
        df.to_excel(w, index=False, sheet_name=table_name)
    out.seek(0)

    return StreamingResponse(
        out,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{table_name}.xlsx"'}
    )

@app.get(
    "/{table_name}/insights",
    summary="Cohere-driven insights for the table"
)
async def insights(
    table_name: str = Path(..., description="Name of the table")
):
    try:
        tables = _get_tables()
        if table_name not in tables:
            raise HTTPException(404, f"Table '{table_name}' not found")

        # reuse fetch_dashboard logic
        with engine.connect() as conn:
            row_count = conn.execute(text(f"SELECT COUNT(*) FROM `{table_name}`")).scalar()
            columns = [r[0] for r in conn.execute(text(f"SHOW COLUMNS FROM `{table_name}`"))]
            sample_df = pd.read_sql_query(text(f"SELECT * FROM `{table_name}` LIMIT 5"), conn)

        prompt = (
            f"You are a data analyst.\n\n"
            f"Table: {table_name}\n"
            f"Rows: {row_count}, Columns: {columns}\n"
            f"Sample (5 rows): {sample_df.to_dict(orient='records')}\n\n"
            "Give me 3 concise insights or anomalies you see."
        )

        resp = await _cohere_generate(prompt)
        raw = resp.generations[0].text.strip()
        insights = [ln.strip() for ln in raw.split("\n") if ln.strip()]
        return {"table": table_name, "insights": insights}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in insights for %s: %s", table_name, e)
        raise HTTPException(status_code=502, detail=str(e))
