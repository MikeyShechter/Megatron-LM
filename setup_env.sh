#!/usr/bin/env bash
# Sets up and enters the Megatron-LM dev environment on JUWELS Booster.
# Uses Apptainer to run the pinned NGC container; no Docker or Pyxis needed.
#
# PREREQUISITE: You must be in the 'container' group.
#   → Sign the Container SLD on JUDOOR, then log out and back in.
#   → Verify with: id | grep container
#
# Workflow (one-time):
#   1. ./setup_env.sh --pull     pull the NGC image to project storage as a .sif
#   2. ./setup_env.sh --setup    create .venv/ and install Python deps inside container
#
# Daily use:
#   ./setup_env.sh               interactive GPU shell on booster
#   ./setup_env.sh --cmd="..."   run a one-off command on a GPU node

set -euo pipefail

# ── Config ─────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_TYPE="dev"
FROM_IMAGE="$(cat "${REPO_ROOT}/docker/.ngc_version.${IMAGE_TYPE}")"

# SIF lives in project storage so it persists long-term (scratch is purged after ~2 months)
SIF="/e/project1/laionize/${USER}/containers/megatron-lm-dev.sif"

# Cache/tmp dirs live in project storage to avoid scratch quota issues.
export APPTAINER_CACHEDIR="/e/project1/laionize/${USER}/container_tmp/APPTAINER_CACHEDIR"
export APPTAINER_TMPDIR="/e/project1/laionize/${USER}/container_tmp/APPTAINER_TMPDIR"

VENV_DIR="${REPO_ROOT}/.venv"
UV_VERSION="0.7.2"            # keep in sync with docker/Dockerfile.ci.dev
DATASETS_DIR="/e/data1/datasets"

# SLURM settings
SLURM_ACCOUNT="reformo"
SLURM_PARTITION="booster"
GPUS_PER_NODE=4               # booster nodes have 4 H200 GPUs each

# ── Argument parsing ────────────────────────────────────────────────────────────
DO_PULL=false
DO_SETUP=false
CUSTOM_CMD=""

for arg in "$@"; do
  case "$arg" in
    --pull)    DO_PULL=true ;;
    --setup)   DO_SETUP=true ;;
    --cmd=*)   CUSTOM_CMD="${arg#--cmd=}" ;;
    --gpus=*)  GPUS_PER_NODE="${arg#--gpus=}" ;;
  esac
done

# Guard: apptainer requires the 'container' group on JUWELS.
# ('singularity' is just a symlink to apptainer — use apptainer directly.)
if ! apptainer --version &>/dev/null; then
  echo "ERROR: apptainer is not available or you are not in the 'container' group."
  echo ""
  echo "  On JUDOOR: Software → Request access to restricted software →"
  echo "  Container Runtime Engine → Get Access → accept the SLD."
  echo "  Then log out and back in. Verify with: id | grep container"
  exit 1
fi

mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

echo "==> Base image : ${FROM_IMAGE}"
echo "==> SIF file   : ${SIF}"
echo "==> venv       : ${VENV_DIR}"
echo ""

# Binds passed to every apptainer exec call
BINDS="${REPO_ROOT}:/workspace,${DATASETS_DIR}:/datasets"

# ── Step 1: Pull the NGC image as a .sif ──────────────────────────────────────
# One-time operation. Re-run only when docker/.ngc_version.dev changes.
# The .sif is stored in project storage so it persists long-term.
if [[ "${DO_PULL}" == "true" ]]; then
  echo "==> Pulling NGC image to ${SIF} ..."
  echo "    This downloads ~15 GB and may take 20-30 min."
  mkdir -p "$(dirname "${SIF}")"
  apptainer pull --force "${SIF}" "docker://${FROM_IMAGE}"
  echo ""
  echo "==> Done. Run ./setup_env.sh --setup next."
  exit 0
fi

# Ensure SIF exists before any run step
if [[ ! -f "${SIF}" ]]; then
  echo "ERROR: SIF not found at ${SIF}."
  echo "  Run ./setup_env.sh --pull first."
  exit 1
fi

# ── Step 2 (first-time): create venv and install Python deps ──────────────────
# Runs uv sync inside the container so the venv inherits system-site-packages
# from the NGC base (torch, CUDA libs, TransformerEngine, NCCL, etc.).
# The venv is written to .venv/ in the repo so every node in a job can see it.
#
# --system-site-packages  inherit torch/cuda/te already in the NGC image
# --locked                pin exactly to uv.lock (reproducibility)
# --no-install-package    skip packages already provided by the NGC base
if [[ "${DO_SETUP}" == "true" ]]; then
  echo "==> Installing Python dependencies inside the container..."
  echo "    (~10-20 min on first run)"
  echo ""

  apptainer exec \
    --bind "${BINDS}" \
    --cleanenv \
    "${SIF}" \
    bash -ex << EOF
    curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh
    export PATH="\${HOME}/.local/bin:\${PATH}"

    export UV_PROJECT_ENVIRONMENT=/workspace/.venv
    export UV_LINK_MODE=copy

    # --system-site-packages makes torch/cuda/te from the NGC image visible
    uv venv /workspace/.venv --system-site-packages

    uv sync --only-group build

    export NVTE_BUILD_NUM_PHILOX_ROUNDS=3
    export NVTE_CUDA_ARCHS="80;90;100"
    uv sync \\
      --extra dev --extra mlm --extra ssm --extra te \\
      --link-mode copy --locked \\
      --no-install-package torch \\
      --no-install-package torchvision \\
      --no-install-package triton \\
      --no-install-package transformer-engine-cu12 \\
      --no-install-package nvidia-cublas-cu12 \\
      --no-install-package nvidia-cuda-cupti-cu12 \\
      --no-install-package nvidia-cuda-nvrtc-cu12 \\
      --no-install-package nvidia-cuda-runtime-cu12 \\
      --no-install-package nvidia-cudnn-cu12 \\
      --no-install-package nvidia-cufft-cu12 \\
      --no-install-package nvidia-cufile-cu12 \\
      --no-install-package nvidia-curand-cu12 \\
      --no-install-package nvidia-cusolver-cu12 \\
      --no-install-package nvidia-cusparse-cu12 \\
      --no-install-package nvidia-cusparselt-cu12 \\
      --no-install-package nvidia-nccl-cu12

    echo "==> Setup complete. Run ./setup_env.sh to enter an interactive shell."
EOF
  exit 0
fi

# ── Step 3: enter / use the container on a GPU node ───────────────────────────
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "ERROR: .venv not found. Run ./setup_env.sh --setup first."
  exit 1
fi

APPTAINER_COMMON="--bind ${BINDS} --nv --cleanenv"

SRUN_COMMON="srun
  --account=${SLURM_ACCOUNT}
  --partition=${SLURM_PARTITION}
  --nodes=1
  --gres=gpu:${GPUS_PER_NODE}
  --ntasks-per-node=1"

if [[ -n "${CUSTOM_CMD}" ]]; then
  ${SRUN_COMMON} \
    apptainer exec ${APPTAINER_COMMON} "${SIF}" \
    bash -c "export PATH=/workspace/.venv/bin:\${PATH}; export VIRTUAL_ENV=/workspace/.venv; ${CUSTOM_CMD}"
else
  echo "==> Requesting interactive shell on ${SLURM_PARTITION} (${GPUS_PER_NODE} GPUs)..."
  # apptainer shell is the standard interactive form per JSC docs
  ${SRUN_COMMON} --pty \
    apptainer shell ${APPTAINER_COMMON} \
    --env "PATH=/workspace/.venv/bin:\$PATH,VIRTUAL_ENV=/workspace/.venv" \
    "${SIF}"
fi
