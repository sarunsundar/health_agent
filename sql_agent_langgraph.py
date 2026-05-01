"""
LangGraph SQL Agent for Health Data — v8 (inject_where Security)

Security model:
  Two-layer row-level security:

  Layer 1 — Unity Catalog Row Filters (Path A: direct users):
    current_user() + is_account_group_member() → email_to_id lookup.
    Works automatically for any user logged into Databricks directly.

  Layer 2 — Application-layer inject_where (Path B: SP/Telegram):
    WHY: Databricks hard constraint — persistent UC functions (row filters)
    CANNOT reference temporary session variables (INVALID_TEMP_OBJ_REFERENCE).
    SOLUTION: inject_where node appends a mandatory WHERE clause to every
    LLM-generated SQL query before execution, binding it to the resolved
    internal_id of the authenticated Telegram user.

    citizen:   WHERE citizen_id  = '{citizen_id}'
    clinician: WHERE clinician_id = '{clinician_id}'

  If neither layer matches → zero rows (safe default).

Features:
  • SQLDatabase.from_uri() for auto schema detection
  • SP OAuth2 with auto-refresh (no PATs)
  • Reflection/retry on SQL execution failure (up to 3 attempts)
  • inject_where for mandatory per-user WHERE clause enforcement
"""

from __future__ import annotations

import os
import re
import time
import logging
import threading
from typing import Optional, Literal, Any

import sqlparse
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger("health_sql_agent")
logging.basicConfig(level=logging.INFO)

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "")
SQL_WAREHOUSE_ID = os.getenv("SQL_WAREHOUSE_ID", "")
SQL_WAREHOUSE_HTTP_PATH = os.getenv("SQL_WAREHOUSE_HTTP_PATH", "")
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct")
CATALOG = os.getenv("UNITY_CATALOG", "")
SCHEMA_CORE = "core"
MAX_RETRIES = 3

# Fully qualified table names
CITIZEN_TABLE = f"{CATALOG}.{SCHEMA_CORE}.citizen"
CLINICIAN_TABLE = f"{CATALOG}.{SCHEMA_CORE}.clinician"
EMAIL_TO_ID_TABLE = f"{CATALOG}.security.email_to_id"
AUDIT_LOG_TABLE = f"{CATALOG}.security.audit_log"
IDENTITY_MAPPING_TABLE = f"{CATALOG}.security.identity_mapping"

# ─────────────────────────────────────────────────────────────────────────────
# Databricks SDK Client (SP OAuth2 auto-refresh)
# ─────────────────────────────────────────────────────────────────────────────

def _get_workspace_client() -> WorkspaceClient:
    """
    Uses DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET from environment.
    SDK auto-generates and refreshes OAuth2 tokens every 1 hour.
    """
    return WorkspaceClient(host=DATABRICKS_HOST)


def _get_sp_token() -> str:
    """Get current SP OAuth2 token for connection strings and LLM calls."""
    w = _get_workspace_client()
    headers = w.config.authenticate()
    return headers.get("Authorization", "").replace("Bearer ", "")


# ─────────────────────────────────────────────────────────────────────────────
# SQLDatabase for Auto Schema Detection
# ─────────────────────────────────────────────────────────────────────────────

def _get_sql_database() -> SQLDatabase:
    """
    Auto-discover table schemas using:
      databricks://token:{token}@{host}?http_path={path}&catalog={cat}&schema={sch}
    """
    token = _get_sp_token()
    host = DATABRICKS_HOST.replace("https://", "").replace("http://", "")
    uri = (
        f"databricks://token:{token}@{host}"
        f"?http_path={SQL_WAREHOUSE_HTTP_PATH}"
        f"&catalog={CATALOG}"
        f"&schema={SCHEMA_CORE}"
    )
    return SQLDatabase.from_uri(
        uri,
        include_tables=["citizen", "clinician"],
        sample_rows_in_table_info=3,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent State
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(BaseModel):
    """LangGraph State flowing through every node."""

    # ── Inputs ──
    prompt: str = Field(description="Natural language query from the user")
    telegram_id: str = Field(description="Telegram user ID")
    bot_type: str = Field(description="'citizen' or 'clinician' — which bot sent the message")

    # ── Resolved identity ──
    databricks_email: Optional[str] = Field(default=None, description="Databricks account email")
    internal_id: Optional[str] = Field(default=None, description="citizen_id or clinician_id")
    role: Optional[Literal["citizen", "clinician"]] = Field(default=None)

    # ── Schema ──
    table_schema: Optional[str] = Field(default=None)

    # ── SQL pipeline ──
    generated_sql: Optional[str] = Field(default=None, description="SQL from LLM")
    sql_is_safe: Optional[bool] = Field(default=None)
    validation_error: Optional[str] = Field(default=None)

    # ── Execution & retry ──
    query_results: Optional[Any] = Field(default=None)
    execution_error: Optional[str] = Field(default=None)
    retry_count: int = Field(default=0)
    previous_errors: list[str] = Field(default_factory=list)

    # ── Output ──
    response: Optional[str] = Field(default=None)
    error: Optional[str] = Field(default=None)


# ─────────────────────────────────────────────────────────────────────────────
# SQL Execution Helper
# ─────────────────────────────────────────────────────────────────────────────

def _execute_statement(statement: str, schema: str = "security"):
    """
    Execute SQL via Databricks SDK (as Service Principal).

    Bug fix: Databricks returns result.result=None when a query succeeds but
    returns 0 rows. The original code treated this as a failure. Now we check
    result.status.state == SUCCEEDED first, then handle empty results correctly.
    """
    w = _get_workspace_client()
    result = w.statement_execution.execute_statement(
        warehouse_id=SQL_WAREHOUSE_ID,
        catalog=CATALOG,
        schema=schema,
        statement=statement,
        wait_timeout="50s",  # increased from 30s to handle warehouse cold-start
    )

    # Check success state first — independent of whether rows were returned
    if result.status and result.status.state == StatementState.SUCCEEDED:
        columns = []
        if result.manifest and result.manifest.schema and result.manifest.schema.columns:
            columns = [col.name for col in result.manifest.schema.columns]
        # result.result is None when 0 rows returned — that is valid, return []
        rows = (result.result.data_array or []) if result.result else []
        return columns, rows

    # Query did not succeed — surface the actual error message
    error_msg = "Query failed"
    if result.status and result.status.error:
        error_msg = result.status.error.message
    elif result.status:
        error_msg = f"Query state: {result.status.state}"
    raise RuntimeError(error_msg)


# ─────────────────────────────────────────────────────────────────────────────
# Audit Logging Helper
# ─────────────────────────────────────────────────────────────────────────────

def _log_audit_event(
    *,
    access_path: str,
    telegram_user_id: str = "",
    databricks_email: str = "",
    role: str = "",
    internal_id: str = "",
    prompt: str = "",
    secured_sql: str = "",
    row_count: Optional[int] = None,
    status: str,
    error_message: str = "",
    duration_ms: int = 0,
) -> None:
    """
    Fire-and-forget audit log insert into health.security.audit_log.

    Runs in a background thread so it never blocks the user response.
    Silently swallows errors — audit failure must not break the main flow.

    Covers Path B (SP/Telegram queries). Path A (direct Databricks users)
    is automatically covered by system.access.audit (Databricks Premium).
    """
    def _insert() -> None:
        try:
            # Escape single-quotes in all string values
            def esc(s: str) -> str:
                return (s or "").replace("'", "''")[:4000]  # cap at 4000 chars

            row_count_sql = str(row_count) if row_count is not None else "NULL"

            sql = (
                f"INSERT INTO {AUDIT_LOG_TABLE} "
                f"(access_path, telegram_user_id, databricks_email, role, internal_id, "
                f" prompt, secured_sql, row_count, status, error_message, duration_ms) "
                f"VALUES ("
                f"'{esc(access_path)}', '{esc(telegram_user_id)}', '{esc(databricks_email)}', "
                f"'{esc(role)}', '{esc(internal_id)}', '{esc(prompt)}', '{esc(secured_sql)}', "
                f"{row_count_sql}, '{esc(status)}', '{esc(error_message)}', {duration_ms})"
            )
            _execute_statement(sql, schema="security")
        except Exception as audit_err:
            logger.warning(f"Audit log insert failed (non-fatal): {audit_err}")

    threading.Thread(target=_insert, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# Node 1: Resolve Identity
# ─────────────────────────────────────────────────────────────────────────────

def resolve_identity(state: AgentState) -> dict:
    """
    Map telegram_id + bot_type → databricks_email + role + internal_id.

    Two lookups:
      1. identity_mapping: telegram_id + bot_type → databricks_email + role
      2. email_to_id: databricks_email + role → citizen_id or clinician_id

    Retry logic: identity lookups are the first queries after a cold warehouse
    start. If the warehouse is waking from sleep, the first query may time out
    even with a 50s wait. We retry up to 2 times with a short delay so the
    warehouse has time to finish starting up.
    """
    logger.info(f"Resolving: telegram_id={state.telegram_id}, bot_type={state.bot_type}")

    max_attempts = 3
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            # Step 1: telegram_id + bot_type → email + role
            _, rows = _execute_statement(
                f"SELECT databricks_email, role "
                f"FROM {IDENTITY_MAPPING_TABLE} "
                f"WHERE telegram_user_id = '{state.telegram_id}' "
                f"  AND bot_type = '{state.bot_type}' "
                f"  AND is_active = TRUE "
                f"LIMIT 1"
            )

            if not rows:
                return {
                    "error": f"Your Telegram account is not registered for the {state.bot_type} bot. "
                             f"Please contact your administrator."
                }

            email, role = rows[0][0], rows[0][1]

            # Step 2: email + role → internal_id (citizen_id or clinician_id)
            id_column = "citizen_id" if role == "citizen" else "clinician_id"
            _, id_rows = _execute_statement(
                f"SELECT {id_column} "
                f"FROM {EMAIL_TO_ID_TABLE} "
                f"WHERE email = '{email}' "
                f"  AND role = '{role}' "
                f"  AND is_active = TRUE "
                f"LIMIT 1"
            )

            if not id_rows or not id_rows[0][0]:
                return {"error": f"No {role} ID found for {email}. Contact your administrator."}

            internal_id = id_rows[0][0]
            logger.info(f"Resolved: {email} → {role}, {id_column}={internal_id}")
            return {"databricks_email": email, "internal_id": internal_id, "role": role}

        except Exception as e:
            last_error = e
            if attempt < max_attempts:
                wait_s = attempt * 5  # 5s after attempt 1, 10s after attempt 2
                logger.warning(
                    f"Identity resolution attempt {attempt}/{max_attempts} failed: {e}. "
                    f"Retrying in {wait_s}s (warehouse may be waking up)..."
                )
                time.sleep(wait_s)
            else:
                logger.error(f"Identity resolution failed after {max_attempts} attempts: {e}")

    return {"error": f"The health database is temporarily unavailable. Please try again in a moment."}


def check_identity(state: AgentState) -> Literal["get_schema", "return_error"]:
    if state.internal_id and state.role:
        return "get_schema"
    return "return_error"


# ─────────────────────────────────────────────────────────────────────────────
# Node 2: Get Schema (Auto-detected)
# ─────────────────────────────────────────────────────────────────────────────

def get_schema(state: AgentState) -> dict:
    """Auto-detect table schemas via SQLDatabase.from_uri()."""
    logger.info("Auto-detecting schema via SQLDatabase.from_uri()")

    try:
        db = _get_sql_database()
        return {"table_schema": db.get_table_info()}
    except Exception as e:
        logger.warning(f"Schema auto-detection failed: {e}")
        fallback = f"""
CREATE TABLE {CITIZEN_TABLE} (
  id BIGINT, citizen_id STRING, name STRING, address STRING,
  health_details STRING, clinician_id STRING
);
CREATE TABLE {CLINICIAN_TABLE} (
  id BIGINT, clinician_id STRING, name STRING, address STRING
);"""
        return {"table_schema": fallback}


# ─────────────────────────────────────────────────────────────────────────────
# Node 3: Generate SQL (Autonomous + Reflection)
# ─────────────────────────────────────────────────────────────────────────────

def generate_sql(state: AgentState) -> dict:
    """
    LLM generates SQL autonomously from natural language.
    
    IMPORTANT: The LLM does NOT handle access control filtering.
    The inject_where node (next step) adds mandatory WHERE clauses
    programmatically. The LLM just generates the base query.
    """
    logger.info(
        f"Generating SQL (attempt {state.retry_count + 1}/{MAX_RETRIES}): "
        f"prompt='{state.prompt}', role={state.role}"
    )

    if state.role == "citizen":
        role_instructions = f"""
The current user is a CITIZEN.
- You may ONLY query `{CITIZEN_TABLE}`.
- Do NOT query `{CLINICIAN_TABLE}`.
- Do NOT add WHERE clauses for access control — that is handled automatically.
- Just write the SELECT query for what the user is asking about.
"""
    else:
        role_instructions = f"""
The current user is a CLINICIAN.
- You may query both `{CITIZEN_TABLE}` and `{CLINICIAN_TABLE}`.
- Do NOT add WHERE clauses for access control — that is handled automatically.
- When user asks about "my patients" → query {CITIZEN_TABLE}.
- When user asks about "my info/profile" → query {CLINICIAN_TABLE}.
"""

    reflection = ""
    if state.previous_errors:
        reflection = "\n\n⚠️ PREVIOUS ATTEMPTS FAILED:\n"
        for i, err in enumerate(state.previous_errors, 1):
            reflection += f"Attempt {i}: {err}\n"
        reflection += "\nGenerate a CORRECTED query.\n"

    system_prompt = f"""You are a Databricks SQL expert.
Convert natural language to valid Databricks SQL.

## Schema
{state.table_schema}

{role_instructions}

RULES:
1. Only SELECT statements. No DML/DDL.
2. Use fully qualified table names: `{CATALOG}.{SCHEMA_CORE}.<table>`.
3. No semicolons.
4. Output ONLY raw SQL — no markdown, no backticks, no explanation.
5. CRITICAL: Always include the `citizen_id` and/or `clinician_id` columns in your SELECT clause, even if the user didn't ask for them. These are required for security verification.
6. IDENTITY FILTERING: DO NOT add any WHERE clauses for `citizen_id` or `clinician_id` yourself. Do NOT attempt to filter by `current_user()`. The security layer injections these filters automatically using your resolved identity. Just write a generic query for the data requested.
{reflection}"""

    try:
        token = _get_sp_token()
        llm = ChatOpenAI(
            model=LLM_ENDPOINT, temperature=0,
            base_url=f"{DATABRICKS_HOST}/serving-endpoints",
            api_key=token,
        )
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=state.prompt),
        ])
        raw = response.content.strip()
        raw = re.sub(r"^```(?:sql)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip().rstrip(";")

        logger.info(f"LLM generated: {raw}")
        return {"generated_sql": raw, "sql_is_safe": None, "validation_error": None}
    except Exception as e:
        logger.error(f"SQL generation failed: {e}")
        return {"error": f"SQL generation error: {str(e)}"}


# ─────────────────────────────────────────────────────────────────────────────
# Node 4: Validate SQL
# ─────────────────────────────────────────────────────────────────────────────

FORBIDDEN_KEYWORDS = {
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE",
    "TRUNCATE", "GRANT", "REVOKE", "MERGE",
}


def validate_sql(state: AgentState) -> dict:
    """Validate: read-only + role-based table access."""
    if state.error:
        return {"sql_is_safe": False, "validation_error": state.error}
    sql = state.generated_sql
    if not sql:
        return {"sql_is_safe": False, "validation_error": "No SQL generated."}

    normalized = sql.strip().upper()
    if not normalized.startswith("SELECT") and not normalized.startswith("WITH"):
        return {"sql_is_safe": False, "validation_error": "Only SELECT/WITH queries allowed."}

    try:
        for stmt in sqlparse.parse(sql):
            for token in stmt.flatten():
                if token.value.upper() in FORBIDDEN_KEYWORDS:
                    return {"sql_is_safe": False, "validation_error": f"Forbidden: {token.value}"}
    except Exception:
        pass

    sql_upper = sql.upper()
    if state.role == "citizen":
        test = re.sub(r'CLINICIAN_ID', '', sql_upper)
        if 'CLINICIAN' in test:
            return {"sql_is_safe": False, "validation_error": "Citizens cannot query clinician table."}

    return {"sql_is_safe": True, "validation_error": None}


def check_sql_safety(state: AgentState) -> Literal["inject_where", "return_error"]:
    if state.sql_is_safe:
        return "inject_where"
    return "return_error"


# ─────────────────────────────────────────────────────────────────────────────
# Node 5: Inject WHERE Clause (Application-layer row security for SP/Telegram)
# ─────────────────────────────────────────────────────────────────────────────

def inject_where(state: AgentState) -> dict:
    """
    Append a mandatory WHERE clause to the LLM-generated SQL.

    WHY THIS EXISTS:
      Databricks persistent functions (Unity Catalog row filters) cannot
      reference temporary session variables (INVALID_TEMP_OBJ_REFERENCE).
      The row filter handles Path A (direct users via current_user()).
      This node handles Path B: SP executing queries on behalf of a
      Telegram user.

    WHAT IT DOES:
      Wraps the LLM query in a subquery and appends a WHERE clause that
      binds results to the resolved internal_id of the authenticated user:

        citizen:   WHERE citizen_id  = '{citizen_id}'
        clinician: WHERE clinician_id = '{clinician_id}'

    SECURITY GUARANTEE:
      The internal_id is resolved in Node 1 (resolve_identity) directly
      from health.security.identity_mapping + email_to_id using SP
      credentials — the LLM never touches or influences it.
      The user cannot bypass or modify this WHERE clause.

    INJECTION STRATEGY:
      Uses a CTE wrapper to safely handle any SELECT/WITH shape:
        WITH _base AS (<original_sql>)
        SELECT * FROM _base WHERE <id_col> = '<safe_id>'

      The safe_id has single-quotes escaped to prevent SQL injection.
    """
    sql = state.generated_sql
    if not sql or not state.internal_id or not state.role:
        return {"error": "Cannot inject WHERE: missing SQL or resolved identity."}

    # Escape single quotes to prevent SQL injection via internal_id
    safe_id = state.internal_id.replace("'", "''")

    if state.role == "citizen":
        id_column = "citizen_id"
    else:
        id_column = "clinician_id"

    # Wrap original query as a CTE, then filter — handles all SELECT/WITH shapes.
    # Requires that the LLM included the id_column in its SELECT list.
    secured_sql = (
        f"WITH _base AS (\n{sql}\n)\n"
        f"SELECT * FROM _base WHERE {id_column} = '{safe_id}'"
    )

    logger.info(
        f"inject_where node: role={state.role}, {id_column}='{state.internal_id}'. "
        f"Wrapped LLM SQL in secured CTE."
    )
    return {"generated_sql": secured_sql}


# ─────────────────────────────────────────────────────────────────────────────
# Node 6: Execute SQL
# ─────────────────────────────────────────────────────────────────────────────

def execute_sql(state: AgentState) -> dict:
    """
    Execute the inject_where-secured SQL as the Service Principal.

    By the time query reaches here, generated_sql has been rewritten by
    inject_where to include a mandatory WHERE clause:
      • citizen:   WHERE citizen_id  = '{resolved_id}'
      • clinician: WHERE clinician_id = '{resolved_id}'

    The Unity Catalog row filter provides an additional defence-in-depth
    layer for any direct Databricks access (Path A users).
    """
    sql_to_run = state.generated_sql
    logger.info(f"Executing: {sql_to_run}")
    access_path = f"telegram_{state.role or 'unknown'}"
    start_ms = int(time.time() * 1000)

    try:
        columns, rows = _execute_statement(sql_to_run, schema=SCHEMA_CORE)
        row_count = len(rows)
        duration_ms = int(time.time() * 1000) - start_ms
        logger.info(f"Returned {row_count} rows in {duration_ms}ms")

        _log_audit_event(
            access_path=access_path,
            telegram_user_id=state.telegram_id,
            databricks_email=state.databricks_email or "",
            role=state.role or "",
            internal_id=state.internal_id or "",
            prompt=state.prompt,
            secured_sql=sql_to_run,
            row_count=row_count,
            status="success",
            duration_ms=duration_ms,
        )
        return {
            "query_results": {"columns": columns, "rows": rows, "row_count": row_count},
            "execution_error": None,
        }
    except Exception as e:
        duration_ms = int(time.time() * 1000) - start_ms
        logger.error(f"Execution failed: {e}")
        _log_audit_event(
            access_path=access_path,
            telegram_user_id=state.telegram_id,
            databricks_email=state.databricks_email or "",
            role=state.role or "",
            internal_id=state.internal_id or "",
            prompt=state.prompt,
            secured_sql=sql_to_run,
            status="error",
            error_message=str(e),
            duration_ms=duration_ms,
        )
        return {"query_results": None, "execution_error": str(e)}


def check_execution(state: AgentState) -> Literal["format_response", "retry_or_fail"]:
    if state.query_results is not None:
        return "format_response"
    return "retry_or_fail"


# ─────────────────────────────────────────────────────────────────────────────
# Node 7: Retry or Fail (Reflection)
# ─────────────────────────────────────────────────────────────────────────────

def retry_or_fail(state: AgentState) -> dict:
    new_count = state.retry_count + 1
    errors = list(state.previous_errors)
    if state.execution_error:
        errors.append(state.execution_error)

    if new_count >= MAX_RETRIES:
        return {
            "retry_count": new_count, "previous_errors": errors,
            "error": f"Query failed after {MAX_RETRIES} attempts. Last: {state.execution_error}",
        }
    return {"retry_count": new_count, "previous_errors": errors}


def check_retry(state: AgentState) -> Literal["generate_sql", "return_error"]:
    if state.retry_count < MAX_RETRIES and not state.error:
        return "generate_sql"
    return "return_error"


# ─────────────────────────────────────────────────────────────────────────────
# Node 8: Format Response
# ─────────────────────────────────────────────────────────────────────────────

def format_response(state: AgentState) -> dict:
    results = state.query_results
    if not results or results["row_count"] == 0:
        return {"response": "No records found matching your query."}

    try:
        token = _get_sp_token()
        llm = ChatOpenAI(
            model=LLM_ENDPOINT, temperature=0,
            base_url=f"{DATABRICKS_HOST}/serving-endpoints", api_key=token,
        )
        
        # Strip system-required ID columns before summarization for a cleaner UX
        all_cols = results["columns"]
        display_indices = [i for i, col in enumerate(all_cols) if col.lower() not in ["citizen_id", "clinician_id"]]
        display_cols = [all_cols[i] for i in display_indices]
        
        table_text = " | ".join(display_cols) + "\n"
        for row in results["rows"][:50]:
            filtered_row = [str(row[i]) if row[i] is not None else "" for i in display_indices]
            table_text += " | ".join(filtered_row) + "\n"

        resp = llm.invoke([
            SystemMessage(content=(
                "Summarize the query results in clear, friendly language. "
                "Do NOT reveal internal IDs or raw column names. "
                "The user is asking about their own health data."
            )),
            HumanMessage(content=f"User asked: \"{state.prompt}\"\n\nResults ({results['row_count']} rows):\n{table_text}"),
        ])
        return {"response": resp.content}
    except Exception as e:
        logger.error(f"Formatting failed: {e}")
        cols = results["columns"]
        lines = [", ".join(f"{c}: {v}" for c, v in zip(cols, r)) for r in results["rows"]]
        return {"response": "Your records:\n" + "\n".join(lines)}


# ─────────────────────────────────────────────────────────────────────────────
# Node 9: Return Error
# ─────────────────────────────────────────────────────────────────────────────

def return_error(state: AgentState) -> dict:
    err = state.error or state.validation_error or "An unknown error occurred."
    # Log blocked/failed attempts for Path B (identity failures, validation blocks)
    if state.telegram_id:  # only log if this was a Telegram request
        status = "blocked" if state.validation_error else "error"
        _log_audit_event(
            access_path=f"telegram_{state.role or 'unknown'}",
            telegram_user_id=state.telegram_id,
            databricks_email=state.databricks_email or "",
            role=state.role or "",
            internal_id=state.internal_id or "",
            prompt=state.prompt,
            secured_sql=state.generated_sql or "",
            status=status,
            error_message=err,
        )
    return {"response": f"Sorry, I couldn't process your request. {err}"}


# ─────────────────────────────────────────────────────────────────────────────
# Graph Assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_agent_graph():
    """
    START → resolve_identity → [ok?]
      ├─ No  → return_error → END
      └─ Yes → get_schema → generate_sql → validate_sql → [safe?]
                               ↑                            │
                               │                       ├─ No  → return_error → END
                               │                       └─ Yes → inject_where → execute_sql → [ok?]
                               │                                                              │
                               │                                              ├─ Yes → format_response → END
                               │                                              └─ No  → retry_or_fail → [retry?]
                               │                                                                         │
                               └────────── Yes ───────────────────────────────────────────────────────────┘
                                           No → return_error → END

    Security enforcement points:
      1. resolve_identity  → telegram_id → email → internal_id (SP credentials)
      2. inject_where      → mandatory WHERE citizen_id/clinician_id = internal_id
      3. UC row filter     → defence-in-depth for direct Databricks users (Path A)
    """
    graph = StateGraph(AgentState)

    graph.add_node("resolve_identity", resolve_identity)
    graph.add_node("get_schema", get_schema)
    graph.add_node("generate_sql", generate_sql)
    graph.add_node("validate_sql", validate_sql)
    graph.add_node("inject_where", inject_where)
    graph.add_node("execute_sql", execute_sql)
    graph.add_node("format_response", format_response)
    graph.add_node("retry_or_fail", retry_or_fail)
    graph.add_node("return_error", return_error)

    graph.add_edge(START, "resolve_identity")
    graph.add_conditional_edges("resolve_identity", check_identity)
    graph.add_edge("get_schema", "generate_sql")
    graph.add_edge("generate_sql", "validate_sql")
    graph.add_conditional_edges("validate_sql", check_sql_safety)
    graph.add_edge("inject_where", "execute_sql")
    graph.add_conditional_edges("execute_sql", check_execution)
    graph.add_conditional_edges("retry_or_fail", check_retry)
    graph.add_edge("format_response", END)
    graph.add_edge("return_error", END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

agent_graph = build_agent_graph()


def query_health_data(prompt: str, telegram_id: str, bot_type: str) -> str:
    """
    Public entry point.
    
    Args:
        prompt: Natural language query.
        telegram_id: Telegram user ID.
        bot_type: 'citizen' or 'clinician'.
    Returns:
        Natural language response.
    """
    result = agent_graph.invoke(AgentState(
        prompt=prompt, telegram_id=telegram_id, bot_type=bot_type,
    ))
    return result["response"]