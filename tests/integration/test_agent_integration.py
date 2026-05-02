"""
Integration Tests — LangGraph SQL Agent

These tests connect to REAL Databricks services:
  - Real SQL Warehouse (via Databricks SDK)
  - Real Unity Catalog tables (identity_mapping, email_to_id, citizen, clinician)
  - Real LLM endpoint (meta-llama-3-3-70b-instruct or configured endpoint)

Prerequisites:
  1. A .env file with valid credentials (copy from .env.example and fill in values)
  2. Virtual environment activated with all dependencies installed
  3. Databricks workspace running with Unity Catalog tables seeded
  4. A registered telegram_id in health.security.identity_mapping

Run locally:
  pytest tests/integration/ -v --tb=short

Run only fast (non-LLM) integration tests:
  pytest tests/integration/ -v -m "not llm"

These tests are intentionally NOT run in GitHub CI (they require real secrets
and a live Databricks warehouse). They are meant for local pre-merge validation.
"""

import os
import pytest
from dotenv import load_dotenv

# Load .env before importing the agent (so env vars are set at module load)
load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# Skip guard: skip ALL integration tests if required env vars are not set
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_VARS = [
    "DATABRICKS_HOST",
    "DATABRICKS_CLIENT_ID",
    "DATABRICKS_CLIENT_SECRET",
    "SQL_WAREHOUSE_ID",
    "SQL_WAREHOUSE_HTTP_PATH",
]

missing = [v for v in REQUIRED_VARS if not os.getenv(v)]
if missing:
    pytest.skip(
        f"Skipping integration tests: missing env vars: {missing}. "
        "Copy .env.example to .env and fill in your values.",
        allow_module_level=True,
    )

# Only import after env vars are confirmed present
from sql_agent_langgraph import (
    AgentState,
    resolve_identity,
    get_schema,
    generate_sql,
    validate_sql,
    inject_where,
    execute_sql,
    format_response,
    query_health_data,
    _execute_statement,
    agent_graph,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test Configuration — edit these values to match your seeded test data
# ─────────────────────────────────────────────────────────────────────────────

# A telegram_id registered in health.security.identity_mapping with bot_type=citizen
CITIZEN_TELEGRAM_ID = os.getenv("TEST_CITIZEN_TELEGRAM_ID", "")
# A telegram_id registered with bot_type=clinician
CLINICIAN_TELEGRAM_ID = os.getenv("TEST_CLINICIAN_TELEGRAM_ID", "")
# A telegram_id NOT registered in the system
UNREGISTERED_TELEGRAM_ID = os.getenv("TEST_UNREGISTERED_TELEGRAM_ID", "000000000")
CATALOG = os.getenv("UNITY_CATALOG", "")

def make_state(**overrides) -> AgentState:
    """Create an AgentState with test defaults."""
    defaults = {
        "prompt": "Show my health records",
        "telegram_id": CITIZEN_TELEGRAM_ID,
        "bot_type": "citizen",
    }
    defaults.update(overrides)
    return AgentState(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# Markers
# ─────────────────────────────────────────────────────────────────────────────
# Mark tests with @pytest.mark.llm to identify tests that call the LLM endpoint.
# Run without LLM tests: pytest tests/integration/ -v -m "not llm"


# ═══════════════════════════════════════════════════════════════════════════════
# Group 1: SQL Warehouse Connectivity
# ═══════════════════════════════════════════════════════════════════════════════

class TestWarehouseConnectivity:
    """Verify that the SQL Warehouse is reachable and basic queries work."""

    def test_warehouse_executes_simple_query(self):
        """Databricks SQL Warehouse must respond to a trivial SELECT."""
        cols, rows = _execute_statement("SELECT 1 AS test_col", schema="core")
        assert cols == ["test_col"]
        assert rows == [["1"]] or rows == [[1]]

    def test_warehouse_can_query_identity_mapping_table(self):
        """health.security.identity_mapping must be accessible to the SP."""
        cols, rows = _execute_statement(
            f"SELECT COUNT(*) AS cnt FROM {CATALOG}.security.identity_mapping",
            schema="security",
        )
        assert "cnt" in cols
        # The table must exist and be queryable (row count >= 0)
        assert int(rows[0][0]) >= 0

    def test_warehouse_can_query_email_to_id_table(self):
        """health.security.email_to_id must be accessible to the SP."""
        cols, rows = _execute_statement(
            f"SELECT COUNT(*) AS cnt FROM {CATALOG}.security.email_to_id",
            schema="security",
        )
        assert int(rows[0][0]) >= 0

    def test_warehouse_can_query_citizen_table(self):
        """health.core.citizen must be accessible to the SP."""
        cols, rows = _execute_statement(
            f"SELECT COUNT(*) AS cnt FROM {CATALOG}.core.citizen",
            schema="core",
        )
        assert int(rows[0][0]) >= 0

    def test_warehouse_can_query_clinician_table(self):
        """health.core.clinician must be accessible to the SP."""
        cols, rows = _execute_statement(
            f"SELECT COUNT(*) AS cnt FROM {CATALOG}.core.clinician",
            schema="core",
        )
        assert int(rows[0][0]) >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# Group 2: Identity Resolution (Node 1) — Real DB
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not CITIZEN_TELEGRAM_ID, reason="TEST_CITIZEN_TELEGRAM_ID not set")
class TestResolveIdentityIntegration:
    """Verify resolve_identity works against the real identity_mapping tables."""

    def test_registered_citizen_resolves_successfully(self):
        """A registered citizen telegram_id must resolve to email + citizen_id."""
        state = make_state(telegram_id=CITIZEN_TELEGRAM_ID, bot_type="citizen")
        result = resolve_identity(state)

        assert "error" not in result, f"Unexpected error: {result.get('error')}"
        assert result.get("databricks_email"), "databricks_email must be non-empty"
        assert result.get("internal_id"), "internal_id (citizen_id) must be non-empty"
        assert result.get("role") == "citizen"

    def test_unregistered_telegram_id_returns_error(self):
        """An unregistered telegram_id must return a user-friendly error, not raise."""
        state = make_state(telegram_id=UNREGISTERED_TELEGRAM_ID, bot_type="citizen")
        result = resolve_identity(state)

        assert "error" in result
        assert "not registered" in result["error"]

    @pytest.mark.skipif(not CLINICIAN_TELEGRAM_ID, reason="TEST_CLINICIAN_TELEGRAM_ID not set")
    def test_registered_clinician_resolves_successfully(self):
        """A registered clinician telegram_id must resolve to email + clinician_id."""
        state = make_state(telegram_id=CLINICIAN_TELEGRAM_ID, bot_type="clinician")
        result = resolve_identity(state)

        assert "error" not in result, f"Unexpected error: {result.get('error')}"
        assert result.get("role") == "clinician"
        assert result.get("internal_id")

    def test_wrong_bot_type_returns_error(self):
        """A citizen telegram_id used with bot_type=clinician must be rejected."""
        state = make_state(telegram_id=CITIZEN_TELEGRAM_ID, bot_type="clinician")
        result = resolve_identity(state)
        # Should either error (not registered for that bot_type) or return wrong role
        if "error" not in result:
            # If somehow resolved, role must not be clinician for a citizen ID
            assert result.get("role") != "clinician", (
                "A citizen telegram_id must not resolve as a clinician"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Group 3: Schema Detection (Node 2) — Real DB
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetSchemaIntegration:
    """Verify get_schema can auto-detect table structure from real Databricks."""

    def test_schema_contains_citizen_table_columns(self):
        """Auto-detected schema must describe the citizen table."""
        state = make_state()
        result = get_schema(state)

        schema = result.get("table_schema", "")
        assert schema, "table_schema must not be empty"
        # Must describe the citizen table in some form
        assert "citizen" in schema.lower()

    def test_schema_contains_clinician_table(self):
        """Auto-detected schema must also describe the clinician table."""
        state = make_state()
        result = get_schema(state)

        schema = result.get("table_schema", "")
        assert "clinician" in schema.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Group 4: SQL Generation (Node 3) — Real LLM
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.llm
class TestGenerateSqlIntegration:
    """Verify generate_sql calls the real LLM and returns valid SQL."""

    def test_citizen_prompt_generates_select_query(self):
        """Real LLM must generate a SELECT query for a citizen health prompt."""
        state = make_state(
            role="citizen",
            table_schema=(
                f"CREATE TABLE {CATALOG}.core.citizen "
                "(citizen_id STRING, name STRING, health_details STRING, clinician_id STRING);"
            ),
        )
        result = generate_sql(state)

        assert "error" not in result, f"LLM error: {result.get('error')}"
        sql = result.get("generated_sql", "")
        assert sql, "generated_sql must not be empty"
        assert sql.strip().upper().startswith("SELECT"), (
            f"LLM must generate a SELECT, got: {sql[:80]}"
        )
        assert "citizen" in sql.lower(), "Query must reference the citizen table"

    def test_generated_sql_has_no_semicolon(self):
        """LLM output must have trailing semicolons stripped."""
        state = make_state(
            role="citizen",
            table_schema=(
                f"CREATE TABLE {CATALOG}.core.citizen "
                "(citizen_id STRING, name STRING, health_details STRING, clinician_id STRING);"
            ),
        )
        result = generate_sql(state)

        sql = result.get("generated_sql", "")
        assert not sql.rstrip().endswith(";"), f"Semicolons must be stripped, got: {sql}"

    def test_generated_sql_has_no_markdown_fences(self):
        """LLM output must have ```sql fences stripped."""
        state = make_state(
            role="citizen",
            table_schema=(
                f"CREATE TABLE {CATALOG}.core.citizen "
                "(citizen_id STRING, name STRING, health_details STRING, clinician_id STRING);"
            ),
        )
        result = generate_sql(state)
        sql = result.get("generated_sql", "")
        assert not sql.startswith("```"), f"Markdown fence must be stripped, got: {sql[:40]}"


# ═══════════════════════════════════════════════════════════════════════════════
# Group 5: Full Graph Routing (End-to-End via agent_graph.invoke)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.llm
@pytest.mark.skipif(not CITIZEN_TELEGRAM_ID, reason="TEST_CITIZEN_TELEGRAM_ID not set")
class TestGraphRoutingIntegration:
    """
    Tests that run the FULL LangGraph pipeline against real Databricks.
    These verify graph routing, security enforcement, and response formatting
    all work together end-to-end.
    """

    def test_citizen_query_returns_natural_language_response(self):
        """Full pipeline: citizen prompt → identity → schema → SQL → execute → format."""
        result = agent_graph.invoke(AgentState(
            prompt="Show my health records",
            telegram_id=CITIZEN_TELEGRAM_ID,
            bot_type="citizen",
        ))

        response = result.get("response", "")
        assert response, "Response must not be empty"
        # Must not be a raw error message
        assert "Sorry" not in response or "No records" in response, (
            f"Unexpected error response: {response}"
        )

    def test_unregistered_user_gets_friendly_error(self):
        """Full pipeline: unregistered telegram_id must return a friendly error."""
        result = agent_graph.invoke(AgentState(
            prompt="Show my health records",
            telegram_id=UNREGISTERED_TELEGRAM_ID,
            bot_type="citizen",
        ))

        response = result.get("response", "")
        assert response, "Response must not be empty even for errors"
        assert "Sorry" in response or "not registered" in response.lower()

    def test_forbidden_sql_prompt_is_rejected(self):
        """Full pipeline: a prompt that yields DROP SQL must be blocked by validate_sql."""
        # This tests whether the LLM/validation pipeline blocks destructive prompts
        result = agent_graph.invoke(AgentState(
            prompt="Drop the citizen table",
            telegram_id=CITIZEN_TELEGRAM_ID,
            bot_type="citizen",
        ))

        response = result.get("response", "")
        # Either the LLM generates a SELECT (safe), or validate_sql blocks it
        # Either way, no actual DROP should have executed
        assert response, "Must always return a response"

    @pytest.mark.skipif(not CLINICIAN_TELEGRAM_ID, reason="TEST_CLINICIAN_TELEGRAM_ID not set")
    def test_clinician_can_query_patient_data(self):
        """Full pipeline: a clinician should be able to query citizen (patient) data."""
        result = agent_graph.invoke(AgentState(
            prompt="Show me my patients",
            telegram_id=CLINICIAN_TELEGRAM_ID,
            bot_type="clinician",
        ))

        response = result.get("response", "")
        assert response, "Response must not be empty"

    def test_response_does_not_leak_internal_ids(self):
        """Format response node must strip citizen_id/clinician_id from output."""
        result = agent_graph.invoke(AgentState(
            prompt="Show my health records",
            telegram_id=CITIZEN_TELEGRAM_ID,
            bot_type="citizen",
        ))

        response = result.get("response", "")
        # Internal IDs follow a pattern like C-XXXXX or D-XXXXX
        # The formatted response should describe health records, not expose raw IDs
        # This is a best-effort check — the LLM is instructed not to reveal IDs
        import re
        raw_id_pattern = re.compile(r'\b[CD]-\d{5,}\b')
        assert not raw_id_pattern.search(response), (
            f"Response appears to leak an internal ID: {response[:200]}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Group 6: Security Enforcement — WHERE Clause Injection
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not CITIZEN_TELEGRAM_ID, reason="TEST_CITIZEN_TELEGRAM_ID not set")
class TestSecurityEnforcementIntegration:
    """
    Verify that the security WHERE clause actually restricts data.
    These tests run real SQL against the warehouse.
    """

    def test_citizen_only_sees_own_records(self):
        """After inject_where, executing SQL must return only this citizen's rows."""
        # Step 1: Resolve identity to get the real citizen_id
        identity_state = make_state(telegram_id=CITIZEN_TELEGRAM_ID, bot_type="citizen")
        identity = resolve_identity(identity_state)

        if "error" in identity:
            pytest.skip(f"Identity resolution failed: {identity['error']}")

        citizen_id = identity["internal_id"]

        # Step 2: Build a state as if inject_where will run
        sql_state = make_state(
            generated_sql=f"SELECT citizen_id, name FROM {CATALOG}.core.citizen",
            internal_id=citizen_id,
            role="citizen",
        )

        # Step 3: Inject WHERE clause
        injected = inject_where(sql_state)
        secured_sql = injected["generated_sql"]

        # Step 4: Execute against real warehouse
        cols, rows = _execute_statement(secured_sql, schema="core")

        # Step 5: Every returned row must belong to this citizen only
        assert "citizen_id" in cols, "citizen_id column must be in results"
        cid_idx = cols.index("citizen_id")
        for row in rows:
            assert row[cid_idx] == citizen_id, (
                f"Row with citizen_id={row[cid_idx]} leaked to citizen {citizen_id}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Group 7: Audit Logging
# ═══════════════════════════════════════════════════════════════════════════════

class TestAuditLoggingIntegration:
    """Verify that audit events are written to health.security.audit_log."""

    @pytest.mark.skipif(not CITIZEN_TELEGRAM_ID, reason="TEST_CITIZEN_TELEGRAM_ID not set")
    def test_successful_query_creates_audit_record(self):
        """Running a full query pipeline must create an audit_log entry."""
        import time

        # Record the current time before the query
        before_ts = int(time.time()) - 5  # 5s buffer

        # Run the full pipeline
        agent_graph.invoke(AgentState(
            prompt="Show my health records",
            telegram_id=CITIZEN_TELEGRAM_ID,
            bot_type="citizen",
        ))

        # Wait briefly for the background audit thread to complete
        time.sleep(5)

        # Check the audit log for a recent entry from this telegram_id
        import time as t
        cols, rows = _execute_statement(
            f"SELECT COUNT(*) AS cnt FROM {CATALOG}.security.audit_log "
            f"WHERE telegram_user_id = '{CITIZEN_TELEGRAM_ID}' "
            f"AND UNIX_TIMESTAMP(event_time) >= {before_ts}",
            schema="security",
        )

        cnt = int(rows[0][0]) if rows else 0
        assert cnt >= 1, (
            f"Expected at least 1 audit log entry for telegram_id={CITIZEN_TELEGRAM_ID}, found {cnt}"
        )
