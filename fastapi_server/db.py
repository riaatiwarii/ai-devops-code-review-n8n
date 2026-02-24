"""
Database layer — PostgreSQL via asyncpg.
Stores review logs, feedback, and metrics.
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional

import asyncpg

from schemas import ReviewOutput

logger = logging.getLogger(__name__)

_pool: Optional[asyncpg.Pool] = None

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/code_reviews",
)

DDL = """
CREATE TABLE IF NOT EXISTS reviews (
    id              SERIAL PRIMARY KEY,
    pr_number       INTEGER NOT NULL,
    repo            VARCHAR(255) NOT NULL,
    diff_hash       VARCHAR(64) NOT NULL,
    issues          JSONB NOT NULL DEFAULT '[]',
    overall_score   NUMERIC(4,2),
    summary         TEXT,
    latency_ms      INTEGER,
    high_severity   BOOLEAN GENERATED ALWAYS AS (
                        jsonb_path_exists(issues, '$[*] ? (@.severity == "high" || @.severity == "critical")')
                    ) STORED,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    human_feedback  JSONB,          -- stores thumbs up/down + notes
    false_positive  BOOLEAN         -- set by human review
);

CREATE INDEX IF NOT EXISTS idx_reviews_repo_pr ON reviews(repo, pr_number);
CREATE INDEX IF NOT EXISTS idx_reviews_created ON reviews(created_at);

CREATE TABLE IF NOT EXISTS system_config (
    key   VARCHAR(100) PRIMARY KEY,
    value TEXT NOT NULL
);

INSERT INTO system_config (key, value)
VALUES ('auto_comment_enabled', 'true'), ('escalation_enabled', 'true')
ON CONFLICT DO NOTHING;
"""


async def init_db():
    global _pool
    _pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    async with _pool.acquire() as conn:
        await conn.execute(DDL)
    logger.info("Database initialized")


async def store_review(
    pr_number: int,
    repo: str,
    diff_hash: str,
    result: ReviewOutput,
    latency_ms: int,
) -> int:
    async with _pool.acquire() as conn:
        row_id = await conn.fetchval(
            """
            INSERT INTO reviews (pr_number, repo, diff_hash, issues, overall_score, summary, latency_ms)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id
            """,
            pr_number,
            repo,
            diff_hash,
            json.dumps([i.model_dump() for i in result.issues]),
            result.overall_score,
            result.summary,
            latency_ms,
        )
    return row_id


async def get_review_by_hash(pr_number: int, repo: str) -> Optional[dict]:
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM reviews WHERE pr_number=$1 AND repo=$2 ORDER BY created_at DESC LIMIT 1",
            pr_number, repo,
        )
    return dict(row) if row else None


async def update_feedback(review_id: int, feedback: dict, false_positive: bool):
    async with _pool.acquire() as conn:
        await conn.execute(
            "UPDATE reviews SET human_feedback=$1, false_positive=$2 WHERE id=$3",
            json.dumps(feedback), false_positive, review_id,
        )


async def get_metrics() -> dict:
    async with _pool.acquire() as conn:
        total = await conn.fetchval("SELECT COUNT(*) FROM reviews")
        high = await conn.fetchval("SELECT COUNT(*) FROM reviews WHERE high_severity = true")
        avg_lat = await conn.fetchval("SELECT AVG(latency_ms) FROM reviews") or 0
        fp_count = await conn.fetchval(
            "SELECT COUNT(*) FROM reviews WHERE false_positive = true"
        ) or 0
        no_issues = await conn.fetchval(
            "SELECT COUNT(*) FROM reviews WHERE jsonb_array_length(issues) = 0"
        ) or 0

    total = total or 1  # avoid division by zero
    return {
        "total": total,
        "high_severity": high,
        "avg_latency_ms": float(avg_lat),
        "auto_cleared_ratio": no_issues / total,
        "escalated_ratio": high / total,
        "false_positive_ratio": fp_count / max(high, 1),
    }


async def get_config(key: str) -> Optional[str]:
    async with _pool.acquire() as conn:
        return await conn.fetchval("SELECT value FROM system_config WHERE key=$1", key)


async def set_config(key: str, value: str):
    async with _pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO system_config (key, value) VALUES ($1, $2) ON CONFLICT (key) DO UPDATE SET value=$2",
            key, value,
        )