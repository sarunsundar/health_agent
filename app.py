"""
FastAPI Application — Databricks App for the Health SQL Agent (v2)

Deployment:
  This app runs as a Databricks App behind the platform's reverse proxy.
  The proxy handles OAuth2 Bearer token validation for all /api/* routes
  BEFORE requests reach this FastAPI code. See:
  https://docs.databricks.com/en/dev-tools/databricks-apps/connect-local.html

Authentication Flow (when deployed as Databricks App):
  1. Caller sends: POST /api/query with "Authorization: Bearer <token>"
  2. Databricks platform reverse proxy validates the token:
     - Checks token is a valid Databricks OAuth2 access token
     - Checks caller has CAN_USE permission on the app
     - Checks token scopes match user_authorization scopes in app.yaml
  3. If invalid → platform returns 401/403 (request NEVER reaches app.py)
  4. If valid   → platform forwards request to this FastAPI app

  Callers include:
    • External apps (OpenClaw/curl): use SP M2M OAuth token
    • Local dev (databricks CLI): use U2M OAuth token
    • Other Databricks apps: use app SP token
    • Databricks notebooks: use audience-scoped token exchange

  Because the platform validates tokens, this app does NOT need to
  re-validate the Authorization header. The only exception is local
  development (running outside Databricks), where SKIP_AUTH=true
  bypasses all auth for convenience.
"""

from __future__ import annotations

import os
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal

from sql_agent_langgraph import query_health_data

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger("health_agent_api")
logging.basicConfig(level=logging.INFO)


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Health SQL Agent API starting up...")
    yield
    logger.info("Health SQL Agent API shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Health Data SQL Agent",
    description=(
        "LangGraph-based SQL Agent for health data. "
        "Converts natural language to SQL with role-based access control, "
        "reflection/retry, and Unity Catalog Row Filter enforcement."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Request body for the /api/query endpoint."""
    prompt: str = Field(
        ...,
        description="Natural language query",
        examples=["Show me my health records"],
    )
    telegram_id: str = Field(
        ...,
        description="Telegram user ID of the requester",
        examples=["444444444"],
    )
    bot_type: Literal["citizen", "clinician"] = Field(
        ...,
        description="Which Telegram bot sent this request",
        examples=["citizen"],
    )


class QueryResponse(BaseModel):
    """Response body for the /api/query endpoint."""
    response: str = Field(description="Natural language response from the agent")
    telegram_id: str = Field(description="Echo of the requesting user's Telegram ID")
    bot_type: str = Field(description="Echo of the bot type")
    processing_time_ms: int = Field(description="Processing time in milliseconds")


class HealthResponse(BaseModel):
    status: str
    version: str


# ─────────────────────────────────────────────────────────────────────────────
# Authentication Guard
# ─────────────────────────────────────────────────────────────────────────────

async def validate_caller_token(request: Request):
    """
    Authentication guard for /api/* endpoints.

    When deployed as a Databricks App:
      The platform reverse proxy validates the caller's OAuth2 Bearer token
      BEFORE the request reaches this function. Invalid tokens are rejected
      with 401/403 at the platform level — they never arrive here.

      Therefore, this function does NOT need to re-validate the token.
      It exists only for:
        1. Local development bypass (SKIP_AUTH=true)
        2. Logging which requests arrive (for audit/debug purposes)

    Reference:
      https://docs.databricks.com/en/dev-tools/databricks-apps/connect-local.html
      "include the Bearer token in the Authorization header" — the platform
      validates this token before forwarding to the app.
    """
    # Local Dev Bypass — when running outside Databricks (e.g., VS Code)
    if os.getenv("SKIP_AUTH", "false").lower() == "true":
        logger.debug("Auth skipped (SKIP_AUTH=true, local development mode)")
        return

    # When deployed on Databricks, the platform already validated the token.
    # Log the request for audit trail purposes.
    logger.info("Request accepted (platform-authenticated)")


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="2.0.0")


@app.post(
    "/api/query",
    response_model=QueryResponse,
    dependencies=[Depends(validate_caller_token)],
    summary="Query health data via natural language",
    description=(
        "Send a prompt, Telegram user ID, and bot_type. "
        "The agent resolves identity, generates SQL (with reflection/retry), "
        "validates, executes, and returns a human-friendly response."
    ),
)
async def query_endpoint(body: QueryRequest):
    """
    Main query endpoint.
    
    Flow:
      1. Receive prompt + telegram_id + bot_type
      2. LangGraph: identity → schema → SQL gen → validate → execute (→ retry) → format
      3. Return natural language response
    """
    logger.info(
        f"Query: telegram_id={body.telegram_id}, bot_type={body.bot_type}, "
        f"prompt='{body.prompt[:80]}...'"
    )

    start = time.time()

    try:
        response_text = query_health_data(
            prompt=body.prompt,
            telegram_id=body.telegram_id,
            bot_type=body.bot_type,
        )
    except Exception as e:
        logger.error(f"Agent error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Agent processing failed: {str(e)}",
        )

    elapsed_ms = int((time.time() - start) * 1000)
    logger.info(f"Query completed in {elapsed_ms}ms")

    return QueryResponse(
        response=response_text,
        telegram_id=body.telegram_id,
        bot_type=body.bot_type,
        processing_time_ms=elapsed_ms,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Local dev
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
