#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

PYTHON_BIN="python"
if [ -x "$PROJECT_DIR/.conda-py310/bin/python" ]; then
    PYTHON_BIN="$PROJECT_DIR/.conda-py310/bin/python"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
elif [ -x "$PROJECT_DIR/.venv-adk/bin/python" ]; then
    PYTHON_BIN="$PROJECT_DIR/.venv-adk/bin/python"
fi

HOST="${SERVICE_HOST:-127.0.0.1}"

pkill -f uvicorn 2>/dev/null || true
sleep 1

# Start all three microservices
"$PYTHON_BIN" -m uvicorn system_agent.server:app --host "$HOST" --port 6000 --log-level warning &
"$PYTHON_BIN" -m uvicorn user_simulator.server:app --host "$HOST" --port 6001 --log-level warning &
"$PYTHON_BIN" -m uvicorn db_environment.server:app --host "$HOST" --port 6002 --log-level warning &

# Wait for all three to be healthy
for i in $(seq 1 30); do
    if curl --noproxy '*' -s "http://127.0.0.1:6000/health" > /dev/null 2>&1 && \
       curl --noproxy '*' -s "http://127.0.0.1:6001/health" > /dev/null 2>&1 && \
       curl --noproxy '*' -s "http://127.0.0.1:6002/health" > /dev/null 2>&1; then
        echo "ALL_SERVICES_READY (ports 6000, 6001, 6002)"
        exit 0
    fi
    sleep 1
done
echo "SERVICES_FAILED"
exit 1
