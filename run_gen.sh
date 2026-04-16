#!/usr/bin/env bash
# =============================================================================
#  LongLive one-click inference dispatcher
# =============================================================================
#  Usage:
#     bash run_gen.sh <seconds> <mode> [ngpus]
#
#  Args:
#     seconds : 10 | 30 | 60 | 120       (video duration in seconds)
#     mode    : base | lora               (base = backbone only, lora = backbone + LoRA)
#     ngpus   : 1-8 (default 1)           single-GPU or multi-GPU
#
#  Examples:
#     bash run_gen.sh  10 base            # 10s, backbone only, 1 GPU
#     bash run_gen.sh  30 lora 8          # 30s, with LoRA, 8 GPUs
#     bash run_gen.sh 120 base 4          # 120s, backbone only, 4 GPUs
#
#  Notes:
#     * All params are inlined; a temp yaml is generated via mktemp and
#       cleaned up on exit.
#     * Prompts are from MovieGenVideoBench.txt (first 128 lines).
#     * Multi-GPU mode isolates CUDA_VISIBLE_DEVICES per rank to prevent
#       ~400MB CUDA context leaking onto GPU 0.
#     * Resume: existing videos in the output dir are skipped automatically.
# =============================================================================

set -euo pipefail

# ---------- Argument validation ----------
if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "Usage: bash $0 <10|30|60|120> <base|lora> [ngpus]"
    exit 1
fi
SEC="$1"
MODE="$2"
NGPUS="${3:-1}"

case "${SEC}" in
    10)  NUM_FRAMES=42  ;;
    30)  NUM_FRAMES=126 ;;
    60)  NUM_FRAMES=252 ;;
    120) NUM_FRAMES=504 ;;
    *) echo "[ERROR] seconds must be one of: 10 30 60 120 (got '${SEC}')"; exit 1 ;;
esac
case "${MODE}" in
    base|lora) ;;
    *) echo "[ERROR] mode must be one of: base lora (got '${MODE}')"; exit 1 ;;
esac
if ! [[ "${NGPUS}" =~ ^[1-8]$ ]]; then
    echo "[ERROR] ngpus must be 1-8 (got '${NGPUS}')"; exit 1
fi

# ---------- Enter script directory ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ---------- Configurable constants ----------
PROMPT_FILE="/home/zdmaogroup/wubin/reward-forcing-claude-add-experiment-runner-script-PPlnc/reward-forcing-claude-spatial-reward-forcing-L5245/LongLive-main/docs/MovieGenVideoBench.txt"
INFERENCE_ITER=127                       # run prompts i=0..127 (128 total)
OUT_DIR="videos/gen_${SEC}s_${MODE}"
GENERATOR_CKPT="longlive_models/models/longlive_base.pt"
LORA_CKPT="longlive_models/models/lora.pt"
LORA_RANK=256
LORA_ALPHA=256

# ---------- Preflight checks ----------
if [ ! -f "${PROMPT_FILE}" ]; then
    echo "[ERROR] prompt file not found: ${PROMPT_FILE}"
    exit 1
fi
NUM_LINES=$(wc -l < "${PROMPT_FILE}")
echo "[INFO] prompt file: ${PROMPT_FILE} (${NUM_LINES} lines, running first 128)"

if [ ! -f "${GENERATOR_CKPT}" ]; then
    echo "[ERROR] generator ckpt not found: ${GENERATOR_CKPT}"
    exit 1
fi
if [ "${MODE}" = "lora" ] && [ ! -f "${LORA_CKPT}" ]; then
    echo "[ERROR] lora ckpt not found (required for mode=lora): ${LORA_CKPT}"
    exit 1
fi
for f in \
    "wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth" \
    "wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
    "wan_models/Wan2.1-T2V-1.3B/config.json"
do
    if [ ! -e "$f" ]; then
        echo "[ERROR] Wan2.1-T2V-1.3B missing file: $f"
        exit 1
    fi
done

mkdir -p "${OUT_DIR}"

# ---------- Generate temp yaml ----------
TMP_YAML="$(mktemp -t longlive_${SEC}s_${MODE}_XXXXXX.yaml)"

cat > "${TMP_YAML}" <<EOF
denoising_step_list:
- 1000
- 750
- 500
- 250
warp_denoising_step: true
num_frame_per_block: 3
model_name: Wan2.1-T2V-1.3B
model_kwargs:
  local_attn_size: 12
  timestep_shift: 5.0
  sink_size: 3

data_path: ${PROMPT_FILE}
output_folder: ${OUT_DIR}
inference_iter: ${INFERENCE_ITER}
num_output_frames: ${NUM_FRAMES}
use_ema: false
seed: 0
num_samples: 1
save_with_index: true
global_sink: true
context_noise: 0

generator_ckpt: ${GENERATOR_CKPT}
EOF

if [ "${MODE}" = "lora" ]; then
    cat >> "${TMP_YAML}" <<EOF
lora_ckpt: ${LORA_CKPT}

adapter:
  type: "lora"
  rank: ${LORA_RANK}
  alpha: ${LORA_ALPHA}
  dropout: 0.0
  dtype: "bfloat16"
  verbose: false
EOF
fi

echo "[INFO] generated temp yaml: ${TMP_YAML}"
echo "----- config -----"
cat "${TMP_YAML}"
echo "------------------"
echo "[INFO] mode   : ${MODE}"
echo "[INFO] length : ${SEC}s  (num_output_frames=${NUM_FRAMES})"
echo "[INFO] gpus   : ${NGPUS}"
echo "[INFO] output : ${OUT_DIR}"
echo "[INFO] resume : existing videos will be skipped"

# ---------- Launch inference ----------
if [ "${NGPUS}" -eq 1 ]; then
    # Single GPU: launch directly
    trap 'rm -f "${TMP_YAML}"' EXIT
    torchrun \
        --nproc_per_node=1 \
        --master_port=29500 \
        inference.py \
        --config_path "${TMP_YAML}"
else
    # Multi-GPU: use a launcher that isolates CUDA_VISIBLE_DEVICES per rank
    # to prevent all ranks from leaking CUDA context onto GPU 0
    TMP_LAUNCHER="$(mktemp -t longlive_launcher_XXXXXX.sh)"
    trap 'rm -f "${TMP_YAML}" "${TMP_LAUNCHER}"' EXIT
    cat > "${TMP_LAUNCHER}" <<'LAUNCH_EOF'
#!/bin/bash
export CUDA_VISIBLE_DEVICES=${LOCAL_RANK}
export LOCAL_RANK=0
exec python -u inference.py "$@"
LAUNCH_EOF
    chmod +x "${TMP_LAUNCHER}"

    torchrun \
        --nproc_per_node="${NGPUS}" \
        --master_port=29500 \
        --no-python \
        "${TMP_LAUNCHER}" --config_path "${TMP_YAML}"
fi

echo "[DONE] ${SEC}s (${MODE}, ${NGPUS}GPU) inference finished, output: ${OUT_DIR}/"
