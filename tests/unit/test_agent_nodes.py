"""
Unit Tests for LangGraph SQL Agent — Node-Level Tests

These tests verify the logic of individual node functions WITHOUT
calling any Databricks services (no LLM, no SQL Warehouse, no network).

All external dependencies are mocked — these tests run 100% locally.

Run: pytest tests/unit/ -v
"""

import pytest
from unittest.mock import patch, MagicMock
from sql_agent_langgraph import (
    AgentState,
    validate_sql,
    inject_where,
    check_identity,
    check_sql_safety,
    check_execution,
    check_retry,
    retry_or_fail,
    return_error,
    FORBIDDEN_KEYWORDS,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: Create an AgentState with defaults
# ═══════════════════════════════════════════════════════════════════════════════

def make_state(**overrides) -> AgentState:
    """Create an AgentState with sensible test defaults."""
    defaults = {
        "prompt": "Show my health records",
        "telegram_id": "12345",
        "bot_type": "citizen",
    }
    defaults.update(overrides)
    return AgentState(**defaults)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: validate_sql node
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidateSql:
    """Tests for the SQL validation node (Node 4)."""

    def test_valid_select_passes(self):
        """A normal SELECT query should pass validation."""
        state = make_state(
            generated_sql="SELECT citizen_id, name FROM health.core.citizen",
            role="citizen",
        )
        result = validate_sql(state)
        assert result["sql_is_safe"] is True
        assert result["validation_error"] is None

    def test_valid_with_cte_passes(self):
        """A WITH (CTE) query should also pass validation."""
        state = make_state(
            generated_sql="WITH cte AS (SELECT * FROM health.core.citizen) SELECT * FROM cte",
            role="citizen",
        )
        result = validate_sql(state)
        assert result["sql_is_safe"] is True

    @pytest.mark.parametrize("keyword", ["DROP", "DELETE", "UPDATE", "INSERT",
                                          "ALTER", "CREATE", "TRUNCATE",
                                          "GRANT", "REVOKE", "MERGE"])
    def test_forbidden_keywords_blocked(self, keyword):
        """All destructive SQL keywords must be blocked."""
        state = make_state(
            generated_sql=f"{keyword} TABLE health.core.citizen",
            role="citizen",
        )
        result = validate_sql(state)
        assert result["sql_is_safe"] is False
        assert "Forbidden" in result["validation_error"] or "Only SELECT" in result["validation_error"]

    def test_drop_table_blocked(self):
        """DROP TABLE must be blocked even in mixed-case."""
        state = make_state(
            generated_sql="Drop Table health.core.citizen",
            role="citizen",
        )
        result = validate_sql(state)
        assert result["sql_is_safe"] is False

    def test_citizen_cannot_query_clinician_table(self):
        """Citizens must not be able to query the clinician table directly."""
        state = make_state(
            generated_sql="SELECT * FROM health.core.clinician",
            role="citizen",
        )
        result = validate_sql(state)
        assert result["sql_is_safe"] is False
        assert "Citizens cannot" in result["validation_error"]

    def test_citizen_can_reference_clinician_id_column(self):
        """Citizens CAN reference clinician_id as a column (it's in citizen table)."""
        state = make_state(
            generated_sql="SELECT citizen_id, clinician_id FROM health.core.citizen",
            role="citizen",
        )
        result = validate_sql(state)
        assert result["sql_is_safe"] is True

    def test_clinician_can_query_both_tables(self):
        """Clinicians should be able to query both citizen and clinician tables."""
        state = make_state(
            generated_sql=(
                "SELECT c.name, cl.name "
                "FROM health.core.citizen c "
                "JOIN health.core.clinician cl ON c.clinician_id = cl.clinician_id"
            ),
            role="clinician",
        )
        result = validate_sql(state)
        assert result["sql_is_safe"] is True

    def test_empty_sql_fails(self):
        """Missing SQL should fail validation."""
        state = make_state(generated_sql=None, role="citizen")
        result = validate_sql(state)
        assert result["sql_is_safe"] is False
        assert "No SQL" in result["validation_error"]

    def test_error_state_propagates(self):
        """If error is already set, validation should fail fast."""
        state = make_state(error="Previous error", generated_sql="SELECT 1", role="citizen")
        result = validate_sql(state)
        assert result["sql_is_safe"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: inject_where node
# ═══════════════════════════════════════════════════════════════════════════════

class TestInjectWhere:
    """Tests for the WHERE clause injection node (Node 5)."""

    def test_citizen_gets_citizen_id_filter(self):
        """Citizen queries must have WHERE citizen_id = '<id>' injected."""
        state = make_state(
            generated_sql="SELECT citizen_id, name FROM health.core.citizen",
            internal_id="C-12345",
            role="citizen",
        )
        result = inject_where(state)
        assert "citizen_id = 'C-12345'" in result["generated_sql"]
        assert "_base" in result["generated_sql"]  # CTE wrapper

    def test_clinician_gets_clinician_id_filter(self):
        """Clinician queries must have WHERE clinician_id = '<id>' injected."""
        state = make_state(
            generated_sql="SELECT clinician_id, name FROM health.core.clinician",
            internal_id="D-67890",
            role="clinician",
            bot_type="clinician",
        )
        result = inject_where(state)
        assert "clinician_id = 'D-67890'" in result["generated_sql"]

    def test_sql_injection_prevention(self):
        """Single quotes in internal_id must be escaped to prevent SQL injection."""
        state = make_state(
            generated_sql="SELECT citizen_id, name FROM health.core.citizen",
            internal_id="C-123'; DROP TABLE citizen; --",
            role="citizen",
        )
        result = inject_where(state)
        # The escaped version should have doubled single quotes
        assert "C-123''; DROP TABLE citizen; --" in result["generated_sql"]
        # The original dangerous string should NOT appear unescaped
        assert "C-123'; DROP" not in result["generated_sql"]

    def test_cte_wrapper_structure(self):
        """Output should wrap original SQL in a CTE: WITH _base AS (...)."""
        state = make_state(
            generated_sql="SELECT citizen_id, name FROM health.core.citizen",
            internal_id="C-001",
            role="citizen",
        )
        result = inject_where(state)
        sql = result["generated_sql"]
        assert sql.startswith("WITH _base AS (")
        assert "SELECT * FROM _base WHERE" in sql

    def test_handles_complex_with_query(self):
        """inject_where should handle queries that already start with WITH."""
        state = make_state(
            generated_sql=(
                "WITH recent AS (SELECT * FROM health.core.citizen WHERE id > 100) "
                "SELECT citizen_id, name FROM recent"
            ),
            internal_id="C-001",
            role="citizen",
        )
        result = inject_where(state)
        assert "citizen_id = 'C-001'" in result["generated_sql"]

    def test_missing_sql_returns_error(self):
        """Should return error if no SQL to inject into."""
        state = make_state(generated_sql=None, internal_id="C-001", role="citizen")
        result = inject_where(state)
        assert "error" in result

    def test_missing_identity_returns_error(self):
        """Should return error if identity not resolved."""
        state = make_state(
            generated_sql="SELECT * FROM health.core.citizen",
            internal_id=None,
            role="citizen",
        )
        result = inject_where(state)
        assert "error" in result


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Routing/Conditional Edge Functions
# ═══════════════════════════════════════════════════════════════════════════════

class TestRoutingFunctions:
    """Tests for conditional edge functions (graph routing logic)."""

    def test_check_identity_success(self):
        """Should route to get_schema when identity is resolved."""
        state = make_state(internal_id="C-001", role="citizen")
        assert check_identity(state) == "get_schema"

    def test_check_identity_failure_no_id(self):
        """Should route to return_error when internal_id is missing."""
        state = make_state(internal_id=None, role="citizen")
        assert check_identity(state) == "return_error"

    def test_check_identity_failure_no_role(self):
        """Should route to return_error when role is missing."""
        state = make_state(internal_id="C-001", role=None)
        assert check_identity(state) == "return_error"

    def test_check_sql_safety_safe(self):
        """Should route to inject_where when SQL is safe."""
        state = make_state(sql_is_safe=True)
        assert check_sql_safety(state) == "inject_where"

    def test_check_sql_safety_unsafe(self):
        """Should route to return_error when SQL is unsafe."""
        state = make_state(sql_is_safe=False)
        assert check_sql_safety(state) == "return_error"

    def test_check_execution_success(self):
        """Should route to format_response when results exist."""
        state = make_state(query_results={"columns": ["id"], "rows": [[1]], "row_count": 1})
        assert check_execution(state) == "format_response"

    def test_check_execution_failure(self):
        """Should route to retry_or_fail when results are None."""
        state = make_state(query_results=None)
        assert check_execution(state) == "retry_or_fail"

    def test_check_retry_can_retry(self):
        """Should route to generate_sql when retries remain."""
        state = make_state(retry_count=1)
        assert check_retry(state) == "generate_sql"

    def test_check_retry_exhausted(self):
        """Should route to return_error when max retries reached."""
        state = make_state(retry_count=3)
        assert check_retry(state) == "return_error"

    def test_check_retry_has_error(self):
        """Should route to return_error when error is set."""
        state = make_state(retry_count=1, error="Some error")
        assert check_retry(state) == "return_error"


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: retry_or_fail node
# ═══════════════════════════════════════════════════════════════════════════════

class TestRetryOrFail:
    """Tests for the retry/fail decision node (Node 7)."""

    def test_increments_retry_count(self):
        state = make_state(retry_count=0, execution_error="Table not found")
        result = retry_or_fail(state)
        assert result["retry_count"] == 1
        assert "Table not found" in result["previous_errors"]

    def test_max_retries_sets_error(self):
        state = make_state(
            retry_count=2,  # Will become 3 (>= MAX_RETRIES)
            execution_error="Still failing",
            previous_errors=["Error 1", "Error 2"],
        )
        result = retry_or_fail(state)
        assert result["retry_count"] == 3
        assert result.get("error") is not None
        assert "3 attempts" in result["error"]

    def test_accumulates_errors(self):
        state = make_state(
            retry_count=0,
            execution_error="New error",
            previous_errors=["Old error"],
        )
        result = retry_or_fail(state)
        assert len(result["previous_errors"]) == 2
        assert "Old error" in result["previous_errors"]
        assert "New error" in result["previous_errors"]


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: return_error node
# ═══════════════════════════════════════════════════════════════════════════════

class TestReturnError:
    """Tests for the error response node (Node 9)."""

    @patch("sql_agent_langgraph._log_audit_event")
    def test_returns_error_message(self, mock_audit):
        state = make_state(error="Database unavailable")
        result = return_error(state)
        assert "Database unavailable" in result["response"]
        assert "Sorry" in result["response"]

    @patch("sql_agent_langgraph._log_audit_event")
    def test_returns_validation_error(self, mock_audit):
        state = make_state(validation_error="Forbidden: DROP")
        result = return_error(state)
        assert "Forbidden: DROP" in result["response"]

    @patch("sql_agent_langgraph._log_audit_event")
    def test_logs_audit_for_telegram_requests(self, mock_audit):
        state = make_state(
            telegram_id="12345",
            error="Test error",
            role="citizen",
        )
        return_error(state)
        mock_audit.assert_called_once()
        call_kwargs = mock_audit.call_args[1]
        assert call_kwargs["telegram_user_id"] == "12345"
        assert call_kwargs["status"] == "error"


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: resolve_identity node (mocked Databricks)
# ═══════════════════════════════════════════════════════════════════════════════

class TestResolveIdentity:
    """Tests for identity resolution node (Node 1) — Databricks calls mocked."""

    @patch("sql_agent_langgraph._execute_statement")
    def test_successful_citizen_resolution(self, mock_exec):
        """Should resolve telegram_id → email → citizen_id."""
        from sql_agent_langgraph import resolve_identity

        # Mock: first call returns identity_mapping row, second returns email_to_id row
        mock_exec.side_effect = [
            (["databricks_email", "role"], [["user@example.com", "citizen"]]),  # identity_mapping
            (["citizen_id"], [["C-12345"]]),  # email_to_id
        ]

        state = make_state(telegram_id="99999", bot_type="citizen")
        result = resolve_identity(state)

        assert result["databricks_email"] == "user@example.com"
        assert result["internal_id"] == "C-12345"
        assert result["role"] == "citizen"

    @patch("sql_agent_langgraph._execute_statement")
    def test_successful_clinician_resolution(self, mock_exec):
        """Should resolve telegram_id → email → clinician_id."""
        from sql_agent_langgraph import resolve_identity

        mock_exec.side_effect = [
            (["databricks_email", "role"], [["doc@example.com", "clinician"]]),
            (["clinician_id"], [["D-67890"]]),
        ]

        state = make_state(telegram_id="88888", bot_type="clinician")
        result = resolve_identity(state)

        assert result["internal_id"] == "D-67890"
        assert result["role"] == "clinician"

    @patch("sql_agent_langgraph._execute_statement")
    def test_unregistered_telegram_id(self, mock_exec):
        """Should return error for unregistered Telegram ID."""
        from sql_agent_langgraph import resolve_identity

        mock_exec.return_value = (["databricks_email", "role"], [])  # No rows

        state = make_state(telegram_id="00000", bot_type="citizen")
        result = resolve_identity(state)

        assert "error" in result
        assert "not registered" in result["error"]

    @patch("sql_agent_langgraph._execute_statement")
    def test_no_internal_id_found(self, mock_exec):
        """Should return error when email exists but no ID mapping."""
        from sql_agent_langgraph import resolve_identity

        mock_exec.side_effect = [
            (["databricks_email", "role"], [["user@example.com", "citizen"]]),
            (["citizen_id"], []),  # No ID rows
        ]

        state = make_state(telegram_id="99999", bot_type="citizen")
        result = resolve_identity(state)

        assert "error" in result
        assert "No citizen ID" in result["error"]


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: generate_sql node (mocked LLM)
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenerateSql:
    """Tests for SQL generation node (Node 3) — LLM calls mocked."""

    @patch("sql_agent_langgraph._get_sp_token", return_value="fake-token")
    @patch("sql_agent_langgraph.ChatOpenAI")
    def test_generates_sql_for_citizen(self, MockLLM, mock_token):
        """Should produce a SELECT query from natural language (citizen)."""
        from sql_agent_langgraph import generate_sql

        mock_response = MagicMock()
        mock_response.content = "SELECT citizen_id, name FROM health.core.citizen"
        MockLLM.return_value.invoke.return_value = mock_response

        state = make_state(
            prompt="Show my health records",
            role="citizen",
            table_schema="CREATE TABLE health.core.citizen (citizen_id STRING, name STRING)",
        )
        result = generate_sql(state)

        assert result["generated_sql"] == "SELECT citizen_id, name FROM health.core.citizen"
        assert result["sql_is_safe"] is None  # Not validated yet

    @patch("sql_agent_langgraph._get_sp_token", return_value="fake-token")
    @patch("sql_agent_langgraph.ChatOpenAI")
    def test_strips_markdown_fences(self, MockLLM, mock_token):
        """Should strip ```sql markdown fences from LLM output."""
        from sql_agent_langgraph import generate_sql

        mock_response = MagicMock()
        mock_response.content = "```sql\nSELECT * FROM health.core.citizen;\n```"
        MockLLM.return_value.invoke.return_value = mock_response

        state = make_state(role="citizen", table_schema="schema text")
        result = generate_sql(state)

        assert not result["generated_sql"].startswith("```")
        assert not result["generated_sql"].endswith(";")

    @patch("sql_agent_langgraph._get_sp_token", return_value="fake-token")
    @patch("sql_agent_langgraph.ChatOpenAI")
    def test_includes_previous_errors_in_reflection(self, MockLLM, mock_token):
        """On retry, previous errors should be included in the LLM prompt."""
        from sql_agent_langgraph import generate_sql

        mock_response = MagicMock()
        mock_response.content = "SELECT citizen_id, name FROM health.core.citizen"
        MockLLM.return_value.invoke.return_value = mock_response

        state = make_state(
            role="citizen",
            table_schema="schema text",
            retry_count=1,
            previous_errors=["Table not found: health.core.citizens"],
        )
        result = generate_sql(state)

        # Verify the LLM was called (the reflection is in the prompt, not the result)
        MockLLM.return_value.invoke.assert_called_once()
        call_args = MockLLM.return_value.invoke.call_args[0][0]
        system_msg = call_args[0].content
        assert "PREVIOUS ATTEMPTS FAILED" in system_msg
        assert "Table not found" in system_msg

    @patch("sql_agent_langgraph._get_sp_token", return_value="fake-token")
    @patch("sql_agent_langgraph.ChatOpenAI")
    def test_llm_error_returns_error(self, MockLLM, mock_token):
        """Should handle LLM API errors gracefully."""
        from sql_agent_langgraph import generate_sql

        MockLLM.return_value.invoke.side_effect = Exception("LLM timeout")

        state = make_state(role="citizen", table_schema="schema text")
        result = generate_sql(state)

        assert "error" in result
        assert "LLM timeout" in result["error"]


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: format_response node (mocked LLM)
# ═══════════════════════════════════════════════════════════════════════════════

class TestFormatResponse:
    """Tests for the response formatting node (Node 8)."""

    def test_empty_results_returns_no_records(self):
        """Should return 'No records found' for empty results."""
        from sql_agent_langgraph import format_response

        state = make_state(
            query_results={"columns": ["name"], "rows": [], "row_count": 0}
        )
        result = format_response(state)
        assert "No records found" in result["response"]

    @patch("sql_agent_langgraph._get_sp_token", return_value="fake-token")
    @patch("sql_agent_langgraph.ChatOpenAI")
    def test_formats_results_with_llm(self, MockLLM, mock_token):
        """Should use LLM to format results into natural language."""
        from sql_agent_langgraph import format_response

        mock_response = MagicMock()
        mock_response.content = "You have 2 health records on file."
        MockLLM.return_value.invoke.return_value = mock_response

        state = make_state(
            query_results={
                "columns": ["citizen_id", "name", "health_details"],
                "rows": [
                    ["C-001", "Alice", "Healthy"],
                    ["C-001", "Alice", "Follow-up needed"],
                ],
                "row_count": 2,
            }
        )
        result = format_response(state)
        assert "2 health records" in result["response"]
