#!/usr/bin/env bash
# =============================================================================
#  LongLive 一键推理分发脚本 (inline params, 临时生成 yaml)
# =============================================================================
#  用法 / Usage:
#     bash run_gen.sh <seconds> <mode>
#
#  参数:
#     seconds : 10 | 30 | 60 | 120       (视频时长, 秒)
#     mode    : base | lora               (base = 纯主干, lora = 主干+LoRA)
#
#  示例:
#     bash run_gen.sh  10 base
#     bash run_gen.sh  30 lora
#     bash run_gen.sh  60 base
#     bash run_gen.sh 120 lora
#
#  说明:
#     * 本脚本不使用任何 configs/ 下的预设 yaml, 所有参数都在这里定义,
#       运行时用 mktemp 临时生成一份 yaml 传给 inference.py, 结束后自动删除.
#     * prompt 取自 docs/MovieGenVideoBench.txt 前 128 条 (inference_iter=127)
#     * 输出目录: videos/gen_<seconds>s_<mode>/
#     * latent frames 按 10s=42 线性缩放:
#         10s -> 42,  30s -> 126,  60s -> 252,  120s -> 504
# =============================================================================

set -euo pipefail

# ---------- 参数校验 ----------
if [ $# -ne 2 ]; then
    echo "Usage: bash $0 <10|30|60|120> <base|lora>"
    exit 1
fi
SEC="$1"
MODE="$2"

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

# ---------- 进入脚本所在目录 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ---------- 可调的固定参数 ----------
PROMPT_FILE="/home/zdmaogroup/wubin/reward-forcing-claude-add-experiment-runner-script-PPlnc/reward-forcing-claude-spatial-reward-forcing-L5245/LongLive-main/docs/MovieGenVideoBench.txt"
INFERENCE_ITER=127                       # 跑 i=0..127 共 128 条
OUT_DIR="videos/gen_${SEC}s_${MODE}"
GENERATOR_CKPT="longlive_models/models/longlive_base.pt"
LORA_CKPT="longlive_models/models/lora.pt"
LORA_RANK=256
LORA_ALPHA=256

# ---------- 前置检查 ----------
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
    echo "[ERROR] lora ckpt not found (mode=lora 必需): ${LORA_CKPT}"
    exit 1
fi
for f in \
    "wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth" \
    "wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
    "wan_models/Wan2.1-T2V-1.3B/config.json"
do
    if [ ! -e "$f" ]; then
        echo "[ERROR] Wan2.1-T2V-1.3B 缺少文件: $f"
        exit 1
    fi
done

mkdir -p "${OUT_DIR}"

# ---------- 临时生成 yaml ----------
TMP_YAML="$(mktemp -t longlive_${SEC}s_${MODE}_XXXXXX.yaml)"
trap 'rm -f "${TMP_YAML}"' EXIT

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
echo "[INFO] output : ${OUT_DIR}"

# ---------- 启动推理 ----------
torchrun \
    --nproc_per_node=1 \
    --master_port=29500 \
    inference.py \
    --config_path "${TMP_YAML}"

echo "[DONE] ${SEC}s (${MODE}) 推理完成, 输出: ${OUT_DIR}/"
