# Source this before any SmashBot work: `source scripts/env.sh`
# Routes all heavy artifacts and caches to drive2. User-level only — no system changes.
export SMASHBOT_DRIVE2=/home/kage/drive2/ShineBot
export UV_PYTHON_INSTALL_DIR=$SMASHBOT_DRIVE2/uv/python
export UV_CACHE_DIR=$SMASHBOT_DRIVE2/uv/cache
export PIP_CACHE_DIR=$SMASHBOT_DRIVE2/uv/pipcache
export HF_HOME=$SMASHBOT_DRIVE2/hf-cache
export WANDB_DIR=$SMASHBOT_DRIVE2/runs
export PATH="$HOME/.local/bin:$PATH"
