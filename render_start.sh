#!/usr/bin/env bash
set -euo pipefail

# Render start helper: write Google credentials (if provided) and launch Uvicorn
# Usage on Render (Python service): Start Command -> `bash render_start.sh`

if [ -n "${GOOGLE_CREDENTIALS_JSON:-}" ]; then
  echo "Writing GOOGLE_CREDENTIALS_JSON to /tmp/google_credentials.json"
  printf "%s" "$GOOGLE_CREDENTIALS_JSON" > /tmp/google_credentials.json
  export GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_credentials.json
fi

# Default port from Render
PORT=${PORT:-8000}

echo "Starting uvicorn on port $PORT"
exec uvicorn src.api:app --host 0.0.0.0 --port "$PORT" --workers 1
