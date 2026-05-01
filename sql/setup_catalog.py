# =============================================================================
# Unity Catalog Setup — Parameterized Python Notebook
#
# WHY PYTHON (not pure SQL with EXECUTE IMMEDIATE):
#   Databricks has two hard SQL constraints that prevent pure-SQL parameterisation of row filter functions:
#
#   1. DECLARE VARIABLE cannot be referenced inside a persistent UC function body 
#     (raises INVALID_TEMP_OBJ_REFERENCE). This is by design — it prevents privilege escalation via session state.
#
#   2. EXECUTE IMMEDIATE with || string concatenation strips the '' escape
#      sequences inside the dynamic string, causing identifiers like 'citizens'
#      to be interpreted as unquoted column names (UNRESOLVED_COLUMN error).
#      Confirmed against official Databricks EXECUTE IMMEDIATE docs:
#      https://docs.databricks.com/en/sql/language-manual/sql-ref-syntax-aux-execute-immediate.html
#
#   SOLUTION: Build the full SQL string in Python (f-strings), then pass the
#   complete, well-formed SQL to spark.sql(). Python substitutes the catalog
#   name BEFORE Databricks sees the SQL — no quoting/escaping issues.
#
# USAGE:
#   Run as a DAB job notebook with a parameter, or manually in a notebook:
#
#   Option A — DAB job (sets parameter automatically):
#     databricks bundle run catalog_setup --target dev
#
#   Option B — Manual notebook run:
#     Set catalog_name and include_test_data widgets at the top of the notebook,
#     or change the DEFAULT values below.
#
# PREREQUISITES (manual, one-time in Account Console):
#   1. Create account-level groups: citizens, clinicians, health_agents, auditors
#   2. Create Service Principals: sp-health-dev, sp-health-prod
# =============================================================================

# Read parameters (DAB job passes these as notebook task parameters)
dbutils.widgets.text("catalog_name", "health_dev", "Target Catalog")
dbutils.widgets.dropdown("include_test_data", "true", ["true", "false"], "Include Test Data")
dbutils.widgets.text("agents_group", "agents_dev", "Agents Group")

catalog = dbutils.widgets.get("catalog_name")
include_test_data = dbutils.widgets.get("include_test_data") == "true"
agents_group = dbutils.widgets.get("agents_group")

spark.sql(f"DROP CATALOG {catalog} CASCADE");

print(f"▶ Setting up catalog: {catalog}")
print(f"▶ Include test data:  {include_test_data}")

# 1. Catalog & Schemas
spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.core")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.security")
print(f"✅ Catalog and schemas created: {catalog}.core, {catalog}.security")

# 2. Core Tables
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {catalog}.core.citizen (
  id             BIGINT    GENERATED ALWAYS AS IDENTITY,
  citizen_id     STRING    NOT NULL,
  name           STRING,
  address        STRING,
  health_details STRING,
  clinician_id   STRING,
  created_at     TIMESTAMP DEFAULT current_timestamp(),
  updated_at     TIMESTAMP DEFAULT current_timestamp()
) USING DELTA
TBLPROPERTIES ('delta.feature.allowColumnDefaults' = 'supported')
COMMENT 'Citizen health records. Row-filtered by Unity Catalog.'""")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {catalog}.core.clinician (
  id           BIGINT    GENERATED ALWAYS AS IDENTITY,
  clinician_id STRING    NOT NULL,
  name         STRING,
  address      STRING,
  created_at   TIMESTAMP DEFAULT current_timestamp(),
  updated_at   TIMESTAMP DEFAULT current_timestamp()
) USING DELTA
TBLPROPERTIES ('delta.feature.allowColumnDefaults' = 'supported')
COMMENT 'Clinician records. Row-filtered by Unity Catalog.'""")

print(f"✅ Core tables created: citizen, clinician")

# 3. Security Tables
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {catalog}.security.identity_mapping (
  telegram_user_id STRING    NOT NULL,
  bot_type         STRING    NOT NULL,
  databricks_email STRING    NOT NULL,
  role             STRING    NOT NULL,
  is_active        BOOLEAN   DEFAULT TRUE,
  created_at       TIMESTAMP DEFAULT current_timestamp()
) USING DELTA
TBLPROPERTIES ('delta.feature.allowColumnDefaults' = 'supported')
COMMENT 'Maps Telegram user IDs + bot type to Databricks account emails.'""")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {catalog}.security.email_to_id (
  email        STRING  NOT NULL,
  role         STRING  NOT NULL,
  citizen_id   STRING,
  clinician_id STRING,
  is_active    BOOLEAN DEFAULT TRUE
) USING DELTA
TBLPROPERTIES ('delta.feature.allowColumnDefaults' = 'supported')
COMMENT 'Maps Databricks account emails to citizen_id/clinician_id for Row Filter.'""")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {catalog}.security.audit_log (
  id               BIGINT    GENERATED ALWAYS AS IDENTITY,
  event_time       TIMESTAMP DEFAULT current_timestamp(),
  access_path      STRING    NOT NULL,
  telegram_user_id STRING,
  databricks_email STRING,
  role             STRING,
  internal_id      STRING,
  prompt           STRING,
  secured_sql      STRING,
  row_count        INT,
  status           STRING    NOT NULL,
  error_message    STRING,
  duration_ms      INT
) USING DELTA
TBLPROPERTIES ('delta.feature.allowColumnDefaults' = 'supported')
COMMENT 'Audit log for SP/Telegram data access (Path B).'""")

print(f"✅ Security tables created: identity_mapping, email_to_id, audit_log")

# 4. Row Filter Functions
#
# WHY PYTHON f-strings (not EXECUTE IMMEDIATE with '' escaping):
#   Databricks strips '' escape sequences when EXECUTE IMMEDIATE evaluates
#   a concatenated string via ||, turning 'citizens' into the unquoted
#   identifier `citizens` which fails with UNRESOLVED_COLUMN.
#
#   Python f-strings substitute {catalog} BEFORE Databricks parses the SQL.
#   The resulting SQL contains proper single-quoted string literals like
#   is_account_group_member('citizens') which work correctly.
#
# Access Paths handled:
#   Path A1 — Citizen logged in directly (current_user() = their email):
#     is_account_group_member('citizens') → TRUE → filter to own rows
#   Path A2 — Clinician logged in directly:
#     is_account_group_member('clinicians') → TRUE → filter to linked patients
#   Path B — Service Principal (health_agents group):
#     Row filter returns TRUE (passthrough) → inject_where in LangGraph
#     provides the mandatory per-user WHERE clause
# =============================================================================

# 4a. Citizen row filter
spark.sql(f"""
CREATE OR REPLACE FUNCTION {catalog}.security.citizen_row_filter(row_citizen_id   STRING, row_clinician_id STRING)
RETURN
  (
    -- Path A1: Citizen logged in directly → own record only
    is_account_group_member('citizens')
    AND row_citizen_id = (
      SELECT citizen_id
      FROM {catalog}.security.email_to_id
      WHERE email = current_user()
        AND role = 'citizen'
        AND is_active = TRUE
      LIMIT 1
    )
  )
  OR
  (
    -- Path A2: Clinician logged in directly → linked patients only
    is_account_group_member('clinicians')
    AND row_clinician_id = (
      SELECT clinician_id
      FROM {catalog}.security.email_to_id
      WHERE email = current_user()
        AND role = 'clinician'
        AND is_active = TRUE
      LIMIT 1
    )
  )
  OR
  (
    -- Path B: Service Principal (health_agents) or Auditor → passthrough
    is_account_group_member('{agents_group}')
    OR is_account_group_member('auditors')
  )""")

# 4b. Clinician row filter
spark.sql(f"""
CREATE OR REPLACE FUNCTION {catalog}.security.clinician_row_filter(row_clinician_id STRING)
RETURN
  (
    -- Path A2: Clinician logged in directly → own record only
    is_account_group_member('clinicians')
    AND row_clinician_id = (
      SELECT clinician_id
      FROM {catalog}.security.email_to_id
      WHERE email = current_user()
        AND role = 'clinician'
        AND is_active = TRUE
      LIMIT 1
    )
  )
  OR
  (
    -- Path B: Service Principal (health_agents) or Auditor → passthrough
    is_account_group_member('{agents_group}')
    OR is_account_group_member('auditors')
  )
""")

print(f"✅ Row filter functions created: citizen_row_filter, clinician_row_filter")

# 5. Apply Row Filters to Tables
spark.sql(f"""ALTER TABLE {catalog}.core.citizen
  SET ROW FILTER {catalog}.security.citizen_row_filter ON (citizen_id, clinician_id)""")

spark.sql(f"""ALTER TABLE {catalog}.core.clinician
  SET ROW FILTER {catalog}.security.clinician_row_filter ON (clinician_id)""")

print(f"✅ Row filters applied to: citizen, clinician")

# 6. Permissions
#    Groups are ACCOUNT-LEVEL — created once, granted per catalog.
grants = [
    # Citizens — can see own citizen record only (enforced by row filter)
    f"GRANT USE CATALOG ON CATALOG {catalog} TO `citizens`",
    f"GRANT USE SCHEMA ON SCHEMA {catalog}.core TO `citizens`",
    f"GRANT SELECT ON TABLE {catalog}.core.citizen TO `citizens`",
    f"GRANT USE SCHEMA ON SCHEMA {catalog}.security TO `citizens`",
    f"GRANT SELECT ON TABLE {catalog}.security.email_to_id TO `citizens`",

    # Clinicians — can see citizen + clinician tables (row filter restricts to linked patients)
    f"GRANT USE CATALOG ON CATALOG {catalog} TO `clinicians`",
    f"GRANT USE SCHEMA ON SCHEMA {catalog}.core TO `clinicians`",
    f"GRANT SELECT ON TABLE {catalog}.core.citizen TO `clinicians`",
    f"GRANT SELECT ON TABLE {catalog}.core.clinician TO `clinicians`",
    f"GRANT USE SCHEMA ON SCHEMA {catalog}.security TO `clinicians`",
    f"GRANT SELECT ON TABLE {catalog}.security.email_to_id TO `clinicians`",

    # Service Principal / health_agents — full read + audit log write (inject_where enforces per-user filtering)
    f"GRANT USE CATALOG ON CATALOG {catalog} TO `{agents_group}`",
    f"GRANT USE SCHEMA ON SCHEMA {catalog}.core TO `{agents_group}`",
    f"GRANT USE SCHEMA ON SCHEMA {catalog}.security TO `{agents_group}`",
    f"GRANT SELECT ON TABLE {catalog}.core.citizen TO `{agents_group}`",
    f"GRANT SELECT ON TABLE {catalog}.core.clinician TO `{agents_group}`",
    f"GRANT SELECT ON TABLE {catalog}.security.identity_mapping TO `{agents_group}`",
    f"GRANT SELECT ON TABLE {catalog}.security.email_to_id TO `{agents_group}`",
    f"GRANT SELECT ON TABLE {catalog}.security.audit_log TO `{agents_group}`",
    f"GRANT MODIFY ON TABLE {catalog}.security.audit_log TO `{agents_group}`"
]

for stmt in grants:
    spark.sql(stmt)

print(f"✅ Grants applied ({len(grants)} statements)")

# 7. Sample / Test Data  (DEV and TEST only — skip for PROD)
if include_test_data:
    spark.sql(f"""
    INSERT INTO {catalog}.core.clinician (clinician_id, name, address) VALUES
      ('D-001', 'Dr. Sarah Lee',  '100 Hospital Blvd, City Health Clinic'),
      ('D-002', 'Dr. James Park', '200 Medical Dr, Heart Care Center'),
      ('D-003', 'Dr. Arun', 'Anna Nagar, Madurai, 625020')""")

    spark.sql(f"""
    INSERT INTO {catalog}.core.citizen
      (citizen_id, name, address, health_details, clinician_id)
    VALUES
      ('C-001', 'Alice Johnson', '123 Main St',
       '{{"bp":"120/80","cholesterol":"normal","diagnoses":["Type 2 Diabetes"],"medications":["Metformin"]}}',
       'D-001'),
      ('C-002', 'Bob Smith', '456 Oak Ave',
       '{{"bp":"130/85","cholesterol":"high","diagnoses":["Hypertension"],"medications":["Atorvastatin"]}}',
       'D-001'),
      ('C-003', 'Carol White', '789 Pine Rd',
       '{{"bp":"118/76","cholesterol":"normal","diagnoses":[],"medications":[]}}',
       'D-002'),
      ('C-004', 'Arun', 'Anna Nagar, North Madurai, 625020','{{"bp":"120/80","cholesterol":"normal"}}', 'D-001'),
      ('C-005', 'Rajesh', 'KK Nagar, North Madurai, 625020', '{{"bp":"130/85","cholesterol":"high","diagnoses": ["Hypertension"]}}', 'D-003'),
      ('C-006', 'Guna', 'Bangalore', '{{"bp":"118/76","cholesterol":"normal"}}', 'D-003')
    """)

    spark.sql(f"""
    INSERT INTO {catalog}.security.email_to_id VALUES
      ('alice@example.com',  'citizen',    'C-001', NULL,    TRUE),
      ('bob@example.com',    'citizen',    'C-002', NULL,    TRUE),
      ('carol@example.com',  'citizen',    'C-003', NULL,    TRUE),
      ('sarah@example.com',  'citizen',    'C-004', NULL,    TRUE),
      ('james@example.com',  'clinician',  NULL,    'D-002', TRUE),
      ('sarunsundar@yahoo.com',  'citizen',    'C-004', NULL,    TRUE),
      ('sarunsundar@yahoo.com',  'clinician',  NULL,    'D-003', TRUE)
    """)

    spark.sql(f"""
    INSERT INTO {catalog}.security.identity_mapping VALUES
      ('111111111', 'citizen',    'alice@example.com',  'citizen',    TRUE, current_timestamp()),
      ('222222222', 'citizen',    'bob@example.com',    'citizen',    TRUE, current_timestamp()),
      ('333333333', 'citizen',    'carol@example.com',  'citizen',    TRUE, current_timestamp()),
      ('444444444', 'citizen',    'sarah@example.com',  'citizen',    TRUE, current_timestamp()),
      ('444444444', 'clinician',  'sarah@example.com',  'clinician',  TRUE, current_timestamp()),
      ('555555555', 'clinician',  'james@example.com',  'clinician',  TRUE, current_timestamp()),
      ('5751557941', 'citizen',    'sarunsundar@yahoo.com',  'citizen',    TRUE, current_timestamp()),
      ('5751557941', 'clinician',  'sarunsundar@yahoo.com',  'clinician',  TRUE, current_timestamp())
    """)

    print(f"✅ Test data inserted into {catalog}")
else:
    print(f"ℹ️  Skipped test data (include_test_data=false) — correct for PROD")

print(f"""
╔══════════════════════════════════════════════════════════╗
  ✅  Catalog setup complete: {catalog}
  • Schemas:    {catalog}.core, {catalog}.security
  • Tables:     citizen, clinician, identity_mapping, email_to_id, audit_log
  • Functions:  citizen_row_filter, clinician_row_filter (row filters applied)
  • Grants:     citizens, clinicians, health_agents, auditors
  • Test data:  {'included' if include_test_data else 'skipped (prod)'}
╚══════════════════════════════════════════════════════════╝
""")
