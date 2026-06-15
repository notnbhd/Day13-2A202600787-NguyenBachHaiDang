#!/usr/bin/env bash
# Local runner for the Observathon sim/score binaries.
# The PyInstaller binary embeds Python 3.12.13 but does NOT bundle `openai` or all of
# its transitive stdlib imports, and it ignores PYTHONPATH (only sys.path[0]=CWD works).
# So we launch from /tmp/orun, which holds: cp312 `openai`+deps, a full 3.12.13 stdlib
# (from `uv python install 3.12`), and a symlink to the repo's telemetry package.
#
# Usage:
#   export OPENAI_API_KEY=<openrouter key>
#   ./run_sim.sh <questions.json> <out.json> [extra sim args...]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN=/tmp/orun
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://openrouter.ai/api/v1}"

QFILE="${1:?questions json}"; OUT="${2:?out json}"; shift 2 || true
# logs/traces should land in the repo, not /tmp/orun
mkdir -p "$REPO/logs"
ln -sfn "$REPO/logs" "$RUN/logs"

cd "$RUN"
"$REPO/observathon-sim" --config "$REPO/solution/config.json" \
  --wrapper "$REPO/solution/wrapper.py" \
  --questions "$QFILE" --out "$OUT" "$@"
