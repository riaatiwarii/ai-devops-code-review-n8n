"""
AI DevOps Code Review - FastAPI Model Server
Exposes inference endpoints for the fine-tuned code review LLM.
"""

import os
import time
import hashlib
import logging
from contextlib import asynccontextmanager
from typing import Optional

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from model import CodeReviewModel
from db import init_db, store_review, get_review_by_hash
from schemas import DiffInput, ReviewOutput, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Globals ────────────────────────────────────────────────────────────────────
model: Optional[CodeReviewModel] = None
redis_client: Optional[redis.Redis] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, redis_client

    logger.info("Loading code review model...")
    model = CodeReviewModel(
        model_name=os.getenv("MODEL_NAME", "deepseek-ai/deepseek-coder-6.7b-instruct"),
        device=os.getenv("DEVICE", "cuda"),
    )
    await model.load()
    logger.info("Model loaded.")

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_client = redis.from_url(redis_url, decode_responses=True)
    logger.info("Redis connected.")

    await init_db()

    yield

    await redis_client.aclose()


app = FastAPI(
    title="AI Code Review API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", 3600))


# ── Helpers ────────────────────────────────────────────────────────────────────

def diff_hash(diff: str) -> str:
    return hashlib.sha256(diff.encode()).hexdigest()


async def get_from_cache(key: str) -> Optional[str]:
    if redis_client:
        return await redis_client.get(f"review:{key}")
    return None


async def set_cache(key: str, value: str):
    if redis_client:
        await redis_client.setex(f"review:{key}", CACHE_TTL, value)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=model is not None and model.is_ready,
        redis_connected=redis_client is not None,
    )


@app.post("/review", response_model=ReviewOutput)
async def review_code(payload: DiffInput, request: Request):
    if not model or not model.is_ready:
        raise HTTPException(503, "Model not ready")

    start = time.time()
    cache_key = diff_hash(payload.diff + payload.language)

    # ── Cache check ────────────────────────────────────────────────────────────
    cached = await get_from_cache(cache_key)
    if cached:
        import json
        result = ReviewOutput(**json.loads(cached))
        result.cached = True
        logger.info(f"Cache HIT for {cache_key[:8]}")
        return result

    # ── Inference ──────────────────────────────────────────────────────────────
    try:
        result: ReviewOutput = await model.review(payload)
    except Exception as e:
        logger.error(f"Model inference failed: {e}")
        raise HTTPException(500, f"Inference error: {str(e)}")

    latency_ms = int((time.time() - start) * 1000)
    result.latency_ms = latency_ms
    result.cached = False

    # ── Store & Cache ──────────────────────────────────────────────────────────
    await set_cache(cache_key, result.model_dump_json())
    await store_review(
        pr_number=payload.pr_number,
        repo=payload.repo,
        diff_hash=cache_key,
        result=result,
        latency_ms=latency_ms,
    )

    logger.info(
        f"Review complete | PR#{payload.pr_number} | "
        f"{len(result.issues)} issues | {latency_ms}ms"
    )
    return result


@app.get("/review/{pr_number}")
async def get_pr_review(pr_number: int, repo: str):
    review = await get_review_by_hash(pr_number=pr_number, repo=repo)
    if not review:
        raise HTTPException(404, "No review found for this PR")
    return review


@app.get("/metrics")
async def metrics():
    """Prometheus-style plaintext metrics endpoint."""
    from db import get_metrics
    m = await get_metrics()
    lines = [
        f"code_review_total {m['total']}",
        f"code_review_high_severity_total {m['high_severity']}",
        f"code_review_avg_latency_ms {m['avg_latency_ms']:.1f}",
        f"code_review_auto_cleared_ratio {m['auto_cleared_ratio']:.3f}",
        f"code_review_escalated_ratio {m['escalated_ratio']:.3f}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)