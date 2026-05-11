#!/bin/bash
#SBATCH --account=reformo
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1        # one task per node; torchrun spawns one process per GPU
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=48         # 4 GPUs × 12 CPUs each
#SBATCH --time=480
#SBATCH --partition=booster
#SBATCH --threads-per-core=1
#SBATCH --job-name=megatron
#SBATCH --output=/e/project1/laionize/shechter1/logs/slurm-%j.out
#SBATCH --error=/e/project1/laionize/shechter1/logs/slurm-%j.out

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch $0 <spec.yaml>"
  exit 2
fi

export CONF=$1

REPO_DIR=/e/project1/laionize/shechter1/repos/Megatron-LM
SIF=/e/project1/laionize/shechter1/containers/megatron-lm-dev.sif
RUN_STORAGE=/e/project1/laionize/shechter1

if [[ "${CONF}" != /* ]]; then
  CONF="${SLURM_SUBMIT_DIR:-$PWD}/${CONF}"
fi
if ! CONF=$(readlink -f "${CONF}"); then
  echo "ERROR: config file not found: $1"
  exit 1
fi
export CONF

CONF_DIR=$(dirname "${CONF}")
APPTAINER_BINDS="${REPO_DIR}:/workspace,/e/data1/datasets:/datasets,${CONF_DIR}:${CONF_DIR},${RUN_STORAGE}:${RUN_STORAGE}"

# W&B — offline so compute nodes don't need outbound network
export WANDB_MODE=offline
export WANDB_DIR=${RUN_STORAGE}/wandb/megatron
export WANDB_CACHE_DIR=${RUN_STORAGE}/.cache/wandb
export WANDB_CONFIG_DIR=${RUN_STORAGE}/.config/wandb
export XDG_CACHE_HOME=${RUN_STORAGE}/.cache
export XDG_CONFIG_HOME=${RUN_STORAGE}/.config
export TMPDIR=${RUN_STORAGE}/tmp/megatron
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$TMPDIR" "${RUN_STORAGE}/logs"

# NCCL / InfiniBand — same as run_moe.sh
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN

# Master node — same resolution pattern as run_moe.sh
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$(getent hosts "$master_addr" | awk '{print $1; exit}')
export MASTER_PORT=$((12802 + ($SLURM_JOBID % 1000)))
export NNODES=$SLURM_NNODES
export GPUS_PER_NODE=4
export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

echo "--- JOB DIAGNOSTICS ---"
echo "Nodes: $NNODES  Total GPUs: $WORLD_SIZE"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Config: $CONF"
echo "-----------------------"

# pretrain_gpt.py now supports --config <spec.yaml>: it reads the flat YAML
# and injects key-value pairs as CLI flags before argparse runs (see arguments.py).
# output_dir and run_name are not Megatron flags; extract them here.
if ! RUN_METADATA=$(
  apptainer exec \
    --bind "${APPTAINER_BINDS}" \
    --cleanenv \
    "${SIF}" \
    /workspace/.venv/bin/python - "${CONF}" <<'PY'
import sys
import yaml

with open(sys.argv[1]) as f:
    spec = yaml.safe_load(f)

print(spec["output_dir"])
print(spec.get("run_name", "megatron_run"))
PY
); then
  echo "ERROR: failed to parse config inside container: ${CONF}"
  exit 1
fi

OUTPUT_DIR=$(sed -n '1p' <<< "${RUN_METADATA}")
RUN_NAME=$(sed -n '2p' <<< "${RUN_METADATA}")

mkdir -p "${OUTPUT_DIR}"

# H200 (Blackwell): CUDA_DEVICE_MAX_CONNECTIONS not required.
# Uncomment for pre-Blackwell with TP>1 or CP>1 (non-FSDP):
# export CUDA_DEVICE_MAX_CONNECTIONS=1

srun \
  --ntasks=${NNODES} \
  --ntasks-per-node=1 \
  --export=ALL \
  apptainer exec \
    --bind "${APPTAINER_BINDS}" \
    --nv \
    "${SIF}" \
  bash -c "
    export PATH=/workspace/.venv/bin:\${PATH}
    export VIRTUAL_ENV=/workspace/.venv
    export TRITON_LIBCUDA_PATH=/.singularity.d/libs
    export LD_LIBRARY_PATH=/.singularity.d/libs:/usr/local/cuda/compat/lib.real:\${LD_LIBRARY_PATH:-}
    NODE_RANK=\${SLURM_NODEID}
    cd /workspace
    python -m torch.distributed.run \
      --nnodes=${NNODES} \
      --nproc-per-node=${GPUS_PER_NODE} \
      --node-rank=\${NODE_RANK} \
      --master-addr=${MASTER_ADDR} \
      --master-port=${MASTER_PORT} \
      pretrain_gpt.py \
        --config \"${CONF}\" \
        --save \"${OUTPUT_DIR}\" \
        --load \"${OUTPUT_DIR}\" \
        --wandb-exp-name \"${RUN_NAME}\"
  "

echo "Job finished."
