# ===== TT-Metal environment =====
export ARCH_NAME=wormhole_b0
export DATA_PATH="/data"
export WORK_HOME="/workspace"
export TT_METAL_HOME="$WORK_HOME/tt-metal-bos"
export PYTHONPATH="$TT_METAL_HOME"
export PYTHON_ENV_DIR="$TT_METAL_HOME/python_env"
export TT_METAL_ENV=dev
export VENDOR=BLACKHOLEPLUS_AS_TT
export TT_METAL_DISABLE_L1_DATA_CACHE_RISCVS="BR,NC,TR,ER"
# Activate TT-Metal python environment
[ -f "$TT_METAL_HOME/python_env/bin/activate" ] && source "$TT_METAL_HOME/python_env/bin/activate"
