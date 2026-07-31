# Source this before any ShineBot work: `source scripts/env.sh`
# Routes all heavy artifacts and caches to drive2. User-level only — no system changes.
export SHINEBOT_DRIVE2=/home/kage/drive2/ShineBot
export UV_PYTHON_INSTALL_DIR=$SHINEBOT_DRIVE2/uv/python
export UV_CACHE_DIR=$SHINEBOT_DRIVE2/uv/cache
export PIP_CACHE_DIR=$SHINEBOT_DRIVE2/uv/pipcache
export HF_HOME=$SHINEBOT_DRIVE2/hf-cache
export WANDB_DIR=$SHINEBOT_DRIVE2/runs
export PATH="$HOME/.local/bin:$PATH"
