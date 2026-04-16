#!/usr/bin/env bash
# =============================================================================
#  LongLive 一键推理分发脚本
# =============================================================================
#  用法 / Usage:
#     bash run_gen.sh <seconds> <mode> [ngpus]
#
#  参数:
#     seconds : 10 | 30 | 60 | 120       (视频时长, 秒)
#     mode    : base | lora               (base = 纯主干, lora = 主干+LoRA)
#     ngpus   : 1-8 (默认 1)              单卡 or 多卡
#
#  示例:
#     bash run_gen.sh  10 base            # 10s, 纯主干, 单卡
#     bash run_gen.sh  30 lora 8          # 30s, 带 LoRA, 八卡
#     bash run_gen.sh 120 base 4          # 120s, 纯主干, 四卡
#
#  说明:
#     * 所有参数内联, 运行时 mktemp 生成临时 yaml, 结束后自动清理
#     * prompt 取自 MovieGenVideoBench.txt 前 128 条
#     * 多卡时每个 rank 的 CUDA_VISIBLE_DEVICES 被隔离到对应物理卡,
#       避免在 GPU 0 上泄漏 ~400MB CUDA context
#     * 支持断点重续: 输出目录下已存在的视频会被自动跳过, 不重复推理
# =============================================================================

set -euo pipefail

# ---------- 参数校验 ----------
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

# ---------- 启动推理 ----------
if [ "${NGPUS}" -eq 1 ]; then
    # 单卡: 直接启动
    trap 'rm -f "${TMP_YAML}"' EXIT
    torchrun \
        --nproc_per_node=1 \
        --master_port=29500 \
        inference.py \
        --config_path "${TMP_YAML}"
else
    # 多卡: 用 launcher 隔离每个 rank 的 CUDA_VISIBLE_DEVICES,
    # 避免所有 rank 在 GPU 0 上泄漏 CUDA context
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

echo "[DONE] ${SEC}s (${MODE}, ${NGPUS}GPU) 推理完成, 输出: ${OUT_DIR}/"
