# Source this file from Bash or Zsh; all caches remain beside this checkout.
if [ -n "${ZSH_VERSION:-}" ]; then
    openchicken_rl_env_source="${(%):-%x}"
else
    openchicken_rl_env_source="${BASH_SOURCE[0]}"
fi
export OPENCHICKEN_RL_ROOT="$(dirname -- "$(readlink -f -- "$openchicken_rl_env_source")")"
unset openchicken_rl_env_source
export OPENCHICKEN_RL_CACHE_ROOT="${OPENCHICKEN_RL_CACHE_ROOT:-$OPENCHICKEN_RL_ROOT/../.cache/external}"
export UV_CACHE_DIR="$OPENCHICKEN_RL_CACHE_ROOT/uv"
export UV_PYTHON_INSTALL_DIR="$OPENCHICKEN_RL_CACHE_ROOT/uv-python"
export UV_PYTHON_BIN_DIR="$OPENCHICKEN_RL_CACHE_ROOT/uv-python-bin"
export UV_PROJECT_ENVIRONMENT="$OPENCHICKEN_RL_ROOT/.venv"
export UV_PYTHON=3.12.13
export UV_HTTP_TIMEOUT=600
export UV_LINK_MODE=copy
export XDG_CACHE_HOME="$OPENCHICKEN_RL_CACHE_ROOT/microduck-runtime"
export XDG_CONFIG_HOME="$OPENCHICKEN_RL_CACHE_ROOT/microduck-config"
export XDG_DATA_HOME="$OPENCHICKEN_RL_CACHE_ROOT/microduck-data"
export HF_HOME="$OPENCHICKEN_RL_CACHE_ROOT/microduck-hf"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export WARP_CACHE_PATH="$XDG_CACHE_HOME/warp"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export WANDB_MODE=offline
export WANDB_DIR="$OPENCHICKEN_RL_ROOT/logs"
export WANDB_CACHE_DIR="$XDG_CACHE_HOME/wandb"
export WANDB_CONFIG_DIR="$XDG_CONFIG_HOME/wandb"
export WANDB_DATA_DIR="$XDG_DATA_HOME/wandb"
export TMPDIR="$OPENCHICKEN_RL_CACHE_ROOT/microduck-tmp"
export MUJOCO_GL=egl
export PYTHONNOUSERSITE=1
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
if [ -f "$UV_PROJECT_ENVIRONMENT/bin/activate" ]; then
    . "$UV_PROJECT_ENVIRONMENT/bin/activate"
fi
