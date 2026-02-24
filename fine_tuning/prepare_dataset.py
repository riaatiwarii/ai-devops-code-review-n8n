"""
Build fine-tuning dataset from GitHub PR history.

Usage:
    python prepare_dataset.py \
        --repos "acme/backend,acme/frontend" \
        --output ./dataset/reviews.jsonl \
        --min_comments 2

This script:
1. Fetches closed PRs with merged status
2. Fetches review comments per PR
3. Matches comments to diff lines
4. Emits JSONL training examples
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}


class GitHubScraper:
    def __init__(self, token: str):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        })
        self.base = "https://api.github.com"

    def _get(self, url: str, params: dict = None) -> dict:
        for attempt in range(3):
            resp = self.session.get(url, params=params)
            if resp.status_code == 429 or (resp.status_code == 403 and "rate limit" in resp.text.lower()):
                wait = int(resp.headers.get("Retry-After", 60))
                logger.warning(f"Rate limited, sleeping {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"Failed after 3 attempts: {url}")

    def _paginate(self, url: str, params: dict = None):
        params = params or {}
        params["per_page"] = 100
        page = 1
        while True:
            params["page"] = page
            data = self._get(url, params)
            if not data:
                break
            yield from data
            page += 1
            if len(data) < 100:
                break

    def get_merged_prs(self, repo: str, max_prs: int = 500):
        url = f"{self.base}/repos/{repo}/pulls"
        count = 0
        for pr in self._paginate(url, {"state": "closed"}):
            if pr.get("merged_at"):
                yield pr
                count += 1
                if count >= max_prs:
                    break

    def get_pr_diff(self, repo: str, pr_number: int) -> str:
        url = f"{self.base}/repos/{repo}/pulls/{pr_number}"
        self.session.headers["Accept"] = "application/vnd.github.v3.diff"
        resp = self.session.get(url)
        self.session.headers["Accept"] = "application/vnd.github.v3+json"
        return resp.text[:8000]  # cap to avoid token limit issues

    def get_review_comments(self, repo: str, pr_number: int) -> list:
        url = f"{self.base}/repos/{repo}/pulls/{pr_number}/comments"
        return list(self._paginate(url))

    def get_pr_files(self, repo: str, pr_number: int) -> list:
        url = f"{self.base}/repos/{repo}/pulls/{pr_number}/files"
        return list(self._paginate(url))


def detect_language(files: list) -> str:
    ext_map = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".go": "go", ".java": "java", ".rs": "rust",
        ".rb": "ruby", ".php": "php", ".cs": "csharp",
    }
    for f in files:
        fname = f.get("filename", "")
        for ext, lang in ext_map.items():
            if fname.endswith(ext):
                return lang
    return "unknown"


def comments_to_issues(comments: list) -> list:
    """Convert GitHub review comments into structured issue format."""
    issues = []
    for c in comments:
        body = c.get("body", "").strip()
        if len(body) < 10:
            continue  # skip trivial comments

        # Simple heuristic severity tagging
        severity = "low"
        if any(kw in body.lower() for kw in ["security", "sql injection", "xss", "auth", "password", "token"]):
            severity = "high"
        elif any(kw in body.lower() for kw in ["bug", "crash", "error", "null", "exception", "fail"]):
            severity = "medium"

        issues.append({
            "file": c.get("path", "unknown"),
            "line": c.get("original_line") or c.get("line") or 0,
            "severity": severity,
            "type": "bug" if severity == "medium" else "security" if severity == "high" else "style",
            "description": body[:300],
            "suggested_fix": None,
            "confidence": 0.75,
        })
    return issues


def build_dataset(repos: list, output_path: str, min_comments: int = 2, max_prs_per_repo: int = 200):
    if not GITHUB_TOKEN:
        raise RuntimeError("Set GITHUB_TOKEN environment variable")

    scraper = GitHubScraper(GITHUB_TOKEN)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with open(output_path, "w") as out:
        for repo in repos:
            logger.info(f"Processing repo: {repo}")
            for pr in scraper.get_merged_prs(repo, max_prs=max_prs_per_repo):
                pr_number = pr["number"]
                try:
                    comments = scraper.get_review_comments(repo, pr_number)
                    if len(comments) < min_comments:
                        continue  # skip PRs with few reviews

                    diff = scraper.get_pr_diff(repo, pr_number)
                    files = scraper.get_pr_files(repo, pr_number)
                    language = detect_language(files)
                    issues = comments_to_issues(comments)

                    example = {
                        "diff": diff,
                        "language": language,
                        "repo": repo,
                        "pr_number": pr_number,
                        "review": {
                            "issues": issues,
                            "overall_score": max(3.0, 10.0 - len(issues) * 0.5),
                            "summary": f"Found {len(issues)} issues in this PR.",
                        },
                    }
                    out.write(json.dumps(example) + "\n")
                    written += 1

                    if written % 50 == 0:
                        logger.info(f"Written {written} examples so far...")

                except Exception as e:
                    logger.warning(f"Error processing PR#{pr_number}: {e}")
                    continue

    logger.info(f"Dataset complete: {written} examples → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repos", required=True, help="Comma-separated owner/repo list")
    parser.add_argument("--output", default="./dataset/reviews.jsonl")
    parser.add_argument("--min_comments", type=int, default=2)
    parser.add_argument("--max_prs", type=int, default=200)
    args = parser.parse_args()

    repos = [r.strip() for r in args.repos.split(",")]
    build_dataset(repos, args.output, args.min_comments, args.max_prs)