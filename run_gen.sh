#!/usr/bin/env bash
# =============================================================================
#  LongLive 一键推理分发脚本
# =============================================================================
#  用法 / Usage:
#     bash run_gen.sh <seconds> <mode>
#
#  参数:
#     seconds : 10 | 30 | 60 | 120       (视频时长, 秒)
#     mode    : base | lora               (base = 纯主干, lora = 主干+LoRA)
#
#  示例:
#     bash run_gen.sh  10 base    # 10s, 纯主干
#     bash run_gen.sh  30 lora    # 30s, 带 LoRA
#     bash run_gen.sh  60 base
#     bash run_gen.sh 120 lora
#
#  说明:
#     * prompt 来自 docs/MovieGenVideoBench.txt 的前 128 条
#     * 所有输出保存到 videos/gen_<seconds>s_<mode>/
#     * latent frames 按 10s=42 线性缩放:
#         10s -> 42,  30s -> 126,  60s -> 252,  120s -> 504
# =============================================================================

set -euo pipefail

# ---------- 参数校验 ----------
if [ $# -ne 2 ]; then
    echo "Usage: bash $0 <10|30|60|120> <base|lora>"
    exit 1
fi
SECONDS_ARG="$1"
MODE_ARG="$2"

case "${SECONDS_ARG}" in
    10|30|60|120) ;;
    *) echo "[ERROR] seconds must be one of: 10 30 60 120 (got '${SECONDS_ARG}')"; exit 1 ;;
esac
case "${MODE_ARG}" in
    base|lora) ;;
    *) echo "[ERROR] mode must be one of: base lora (got '${MODE_ARG}')"; exit 1 ;;
esac

CONFIG_PATH="configs/longlive_inference_${SECONDS_ARG}s_${MODE_ARG}.yaml"
OUT_DIR="videos/gen_${SECONDS_ARG}s_${MODE_ARG}"

if [ ! -f "${CONFIG_PATH}" ]; then
    echo "[ERROR] config not found: ${CONFIG_PATH}"
    exit 1
fi

# ---------- 进入脚本所在目录 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ---------- 确认 prompt 文件 ----------
PROMPT_FILE="/home/zdmaogroup/wubin/reward-forcing-claude-add-experiment-runner-script-PPlnc/reward-forcing-claude-spatial-reward-forcing-L5245/LongLive-main/docs/MovieGenVideoBench.txt"
if [ ! -f "${PROMPT_FILE}" ]; then
    echo "[ERROR] prompt file not found: ${PROMPT_FILE}"
    exit 1
fi
NUM_LINES=$(wc -l < "${PROMPT_FILE}")
echo "[INFO] prompt file: ${PROMPT_FILE} (${NUM_LINES} lines, running first 128)"

# ---------- 权重自检 ----------
if [ ! -f "longlive_models/models/longlive_base.pt" ]; then
    echo "[ERROR] longlive_models/models/longlive_base.pt not found"
    echo "        请先下载 LongLive-1.3B 主干权重."
    exit 1
fi
if [ "${MODE_ARG}" = "lora" ] && [ ! -f "longlive_models/models/lora.pt" ]; then
    echo "[ERROR] longlive_models/models/lora.pt not found (mode=lora 时必需)"
    echo "        请先下载 LongLive-1.3B 的 LoRA 权重."
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

# ---------- 启动推理 ----------
mkdir -p "${OUT_DIR}"
echo "[INFO] config : ${CONFIG_PATH}"
echo "[INFO] output : ${OUT_DIR}"
echo "[INFO] mode   : ${MODE_ARG}"
echo "[INFO] length : ${SECONDS_ARG}s"

torchrun \
    --nproc_per_node=1 \
    --master_port=29500 \
    inference.py \
    --config_path "${CONFIG_PATH}"

echo "[DONE] ${SECONDS_ARG}s (${MODE_ARG}) 推理完成, 输出: ${OUT_DIR}/"
