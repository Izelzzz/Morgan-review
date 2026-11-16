#!/bin/bash
export UV_SYSTEM_CERTS=1; unset UV_NATIVE_TLS
export REVIEW_AGENTS_DATA_DIR=/tmp/claude-0/-home-user/a2103717-d4fe-51c2-94ba-df7e29a4e5fc/scratchpad/smoke/data-eval1
export REVIEW_AGENTS_SHARED_SECRET=$(cat /tmp/claude-0/-home-user/a2103717-d4fe-51c2-94ba-df7e29a4e5fc/scratchpad/smoke/secret)
export REVIEW_AGENTS_FINDER_MODEL__BASE_URL=https://api.deepseek.com
export REVIEW_AGENTS_FINDER_MODEL__MODEL=deepseek-flash
export REVIEW_AGENTS_FINDER_MODEL__API_KEY=proxy-injects-the-real-key
export REVIEW_AGENTS_VERIFIER_MODEL__BASE_URL=https://api.deepseek.com
export REVIEW_AGENTS_VERIFIER_MODEL__MODEL=deepseek-flash
export REVIEW_AGENTS_VERIFIER_MODEL__API_KEY=proxy-injects-the-real-key
cd /home/user/review-lab-wt/main
export REVIEW_AGENTS_STRATEGY=walkthrough_first
exec uv run python -m review_agents
