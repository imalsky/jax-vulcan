#!/usr/bin/env bash
# ====== RELEASE + UPLOAD (TestPyPI) + REINSTALL (vulcan) ======
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$SCRIPT_DIR"
LOG="${HOME}/Desktop/vulcan_jax_testpypi_release_$(date +'%Y-%m-%d_%H%M%S').log"

exec > >(tee -a "$LOG") 2>&1

echo "Logging to: $LOG"
echo "Shell: $SHELL"
echo "PWD (start): $(pwd)"

trap 'echo "ERROR: command failed (exit=$?) at line $LINENO. See log: $LOG"' ERR

cd "$REPO"

which python || true
python -V || true
python -m pip -V || true

# 1) Thorough clean
rm -rf dist build .eggs
rm -rf .pytest_cache .mypy_cache .ruff_cache
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete
find . -name "*.egg-info" -prune -exec rm -rf {} +
find . -name ".DS_Store" -delete

# 2) Bump patch version in src/vulcan_jax/_version.py
VER="$(
python - <<'PY'
from pathlib import Path
import re

p = Path("src/vulcan_jax/_version.py")
t = p.read_text(encoding="utf-8")

m = re.search(r'__version__\s*=\s*"(\d+)\.(\d+)\.(\d+)"', t)
if not m:
    raise SystemExit("ERROR: could not parse __version__ in src/vulcan_jax/_version.py")

a, b, c = map(int, m.groups())
new = f"{a}.{b}.{c+1}"

p.write_text(
    re.sub(r'__version__\s*=\s*"\d+\.\d+\.\d+"', f'__version__ = "{new}"', t),
    encoding="utf-8",
)
print(new)
PY
)"
echo "Releasing version: $VER"

# 3) Build first to catch errors before committing
python -m pip install --upgrade pip build twine
python -m build

# 4) Commit + push (only after build succeeds)
git status
git add -u
git add src/vulcan_jax/_version.py
git commit -m "Release v$VER"
git push origin main

export TWINE_USERNAME="__token__"
python -m twine upload --verbose --repository-url https://test.pypi.org/legacy/ dist/*

# 5) Reinstall into vulcan conda env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vulcan
python -V
python -m pip uninstall -y vulcan-jax || true

echo "Waiting for TestPyPI to index v$VER ..."
for i in $(seq 1 12); do
  sleep 10
  if python -m pip install --dry-run --no-cache-dir \
       --index-url https://test.pypi.org/simple/ \
       --extra-index-url https://pypi.org/simple/ \
       "vulcan-jax==$VER" 2>/dev/null; then
    echo "v$VER is available!"
    break
  fi
  echo "  attempt $i/12 -- not indexed yet, retrying..."
done

python -m pip install --no-cache-dir \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  "vulcan-jax==$VER"

python -c "import vulcan_jax; print('installed', vulcan_jax.__version__)"

echo "DONE. Full transcript: $LOG"
