#!/usr/bin/env bash
# =============================================================================
#  LongLive one-click inference script: download weights + generate 180s video
# =============================================================================
#  用法 / Usage:
#     bash run_180s.sh
#
#  说明 / Notes:
#   * 通过 https://hf-mirror.com 镜像下载 Wan2.1-T2V-1.3B 与 LongLive-1.3B 权重
#     - Wan2.1-T2V-1.3B 提供 T5 文本编码器 / VAE / tokenizer / 架构 config
#     - LongLive-1.3B 提供微调后的 generator 权重和 LoRA
#   * 只用短片训练得到的主干 longlive_base.pt, **不加载 LoRA**
#   * 生成一个约 180 秒的长视频（16fps * 180s = 2880 帧 ≈ 720 latent frames）
#   * 结果保存在 videos/long_180s/ 目录下
#
#  注意: 1) 主干本身只在短片 (~5s) 上训过, 直接推 720 latent frames 大概率会
#           在后半段出现分布漂移/崩坏/重复, 这是预期行为, 不是 bug.
#        2) 180s 长视频对显存要求更高, 如爆显存可在 yaml 中调小 num_output_frames
#           (需保持能被 num_frame_per_block=3 整除).
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

# ---------- 4. 下载 Wan2.1-T2V-1.3B 基座 (T5 / VAE / tokenizer / config) ----------
WAN_DIR="wan_models/Wan2.1-T2V-1.3B"
if [ ! -f "${WAN_DIR}/Wan2.1_VAE.pth" ] || \
   [ ! -f "${WAN_DIR}/models_t5_umt5-xxl-enc-bf16.pth" ] || \
   [ ! -f "${WAN_DIR}/config.json" ]; then
    echo "[INFO] 下载 Wan-AI/Wan2.1-T2V-1.3B 到 ${WAN_DIR} ..."
    mkdir -p "${WAN_DIR}"
    huggingface-cli download \
        Wan-AI/Wan2.1-T2V-1.3B \
        --local-dir "${WAN_DIR}" \
        --local-dir-use-symlinks False \
        --resume-download
else
    echo "[INFO] Wan2.1-T2V-1.3B 已存在，跳过下载"
fi

# ---------- 5. 下载 LongLive-1.3B 主干 + 提示词 (不需要 lora.pt) ----------
LONGLIVE_DIR="longlive_models"
if [ ! -f "${LONGLIVE_DIR}/models/longlive_base.pt" ] || \
   [ ! -f "${LONGLIVE_DIR}/prompts/vidprom_filtered_extended.txt" ]; then
    echo "[INFO] 下载 Efficient-Large-Model/LongLive-1.3B 到 ${LONGLIVE_DIR} ..."
    mkdir -p "${LONGLIVE_DIR}"
    huggingface-cli download \
        Efficient-Large-Model/LongLive-1.3B \
        --local-dir "${LONGLIVE_DIR}" \
        --local-dir-use-symlinks False \
        --resume-download \
        --include "models/longlive_base.pt" "prompts/*"
else
    echo "[INFO] LongLive-1.3B 主干已存在，跳过下载"
fi

# ---------- 6. 确认提示词文件存在 ----------
PROMPT_FILE="${LONGLIVE_DIR}/prompts/vidprom_filtered_extended.txt"
if [ ! -f "${PROMPT_FILE}" ]; then
    echo "[WARN] 未找到官方提示词文件 ${PROMPT_FILE}"
    echo "[WARN] 将使用 example/long_example.txt 作为备用 data_path"
    PROMPT_FILE="example/long_example.txt"
fi
echo "[INFO] 使用提示词文件: ${PROMPT_FILE}"

# ---------- 7. 运行 180s 视频推理 ----------
CONFIG_PATH="configs/longlive_inference_180s.yaml"
echo "[INFO] 启动推理: ${CONFIG_PATH}"
echo "[INFO] 输出目录 : videos/long_180s"

mkdir -p videos/long_180s

torchrun \
    --nproc_per_node=1 \
    --master_port=29500 \
    inference.py \
    --config_path "${CONFIG_PATH}"

echo "[DONE] 180 秒视频已生成, 请查看 videos/long_180s/ 目录"
