#!/usr/bin/env bash
# =============================================================================
#  LongLive one-click inference script: download LongLive weights + 120s video
# =============================================================================
#  用法 / Usage:
#     bash run_120s.sh
#
#  说明 / Notes:
#   * 通过 https://hf-mirror.com 镜像下载 LongLive-1.3B 权重
#   * 生成一个约 120 秒的长视频（16fps * 120s = 1920 帧 ≈ 480 latent frames）
#   * 结果保存在 videos/long_120s/ 目录下
#
#  注意: 本脚本只下载推理微调权重 (LongLive-1.3B)。
#        代码中 utils/wan_wrapper.py 仍然会从 wan_models/Wan2.1-T2V-1.3B/
#        读取 T5 文本编码器 / VAE / tokenizer / 架构 config，
#        请自行确保以下文件存在 (若没有请自行准备):
#           wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth
#           wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
#           wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/
#           wan_models/Wan2.1-T2V-1.3B/config.json  (以及配套 safetensors)
# =============================================================================

set -euo pipefail

# ---------- 1. 进入脚本所在目录 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ---------- 2. 配置 HuggingFace 镜像 ----------
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

echo "[INFO] HF_ENDPOINT = ${HF_ENDPOINT}"

# ---------- 3. 确认 huggingface-cli 可用 ----------
if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "[INFO] huggingface-cli 未找到，正在安装 huggingface_hub[cli]..."
    pip install -U "huggingface_hub[cli]" hf_transfer
fi

# ---------- 4. 下载 LongLive-1.3B 权重 + 提示词 ----------
LONGLIVE_DIR="longlive_models"
if [ ! -f "${LONGLIVE_DIR}/models/longlive_base.pt" ] || \
   [ ! -f "${LONGLIVE_DIR}/models/lora.pt" ] || \
   [ ! -f "${LONGLIVE_DIR}/prompts/vidprom_filtered_extended.txt" ]; then
    echo "[INFO] 下载 Efficient-Large-Model/LongLive-1.3B 到 ${LONGLIVE_DIR} ..."
    mkdir -p "${LONGLIVE_DIR}"
    huggingface-cli download \
        Efficient-Large-Model/LongLive-1.3B \
        --local-dir "${LONGLIVE_DIR}" \
        --local-dir-use-symlinks False \
        --resume-download
else
    echo "[INFO] LongLive-1.3B 已存在，跳过下载"
fi

# ---------- 5. 前置检查: Wan 基座必要文件 ----------
WAN_DIR="wan_models/Wan2.1-T2V-1.3B"
MISSING=0
for f in \
    "${WAN_DIR}/models_t5_umt5-xxl-enc-bf16.pth" \
    "${WAN_DIR}/Wan2.1_VAE.pth" \
    "${WAN_DIR}/google/umt5-xxl" \
    "${WAN_DIR}/config.json"
do
    if [ ! -e "$f" ]; then
        echo "[ERROR] 缺少文件: $f"
        MISSING=1
    fi
done
if [ "$MISSING" -ne 0 ]; then
    echo "[ERROR] 推理需要 Wan2.1-T2V-1.3B 的 T5 / VAE / tokenizer / config 文件,"
    echo "        请先放置到 ${WAN_DIR}/ 后再运行本脚本."
    exit 1
fi

# ---------- 6. 确认提示词文件存在 ----------
PROMPT_FILE="${LONGLIVE_DIR}/prompts/vidprom_filtered_extended.txt"
if [ ! -f "${PROMPT_FILE}" ]; then
    echo "[WARN] 未找到官方提示词文件 ${PROMPT_FILE}"
    echo "[WARN] 将使用 example/long_example.txt 作为备用 data_path"
    PROMPT_FILE="example/long_example.txt"
fi
echo "[INFO] 使用提示词文件: ${PROMPT_FILE}"

# ---------- 7. 运行 120s 视频推理 ----------
CONFIG_PATH="configs/longlive_inference_120s.yaml"
echo "[INFO] 启动推理: ${CONFIG_PATH}"
echo "[INFO] 输出目录 : videos/long_120s"

mkdir -p videos/long_120s

torchrun \
    --nproc_per_node=1 \
    --master_port=29500 \
    inference.py \
    --config_path "${CONFIG_PATH}"

echo "[DONE] 120 秒视频已生成, 请查看 videos/long_120s/ 目录"
