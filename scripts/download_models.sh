#!/bin/bash
#
# FunASR C++ Runtime 模型下载脚本
# 从 ModelScope 下载所有 ONNX 模型 + CAM++ PyTorch 模型
# 目录结构: model_zoo/功能/模型名/框架
#

set -e

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_ZOO="${BASE_DIR}/model_zoo"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 检查 modelscope CLI
if ! command -v modelscope &> /dev/null; then
    log_error "modelscope CLI 未安装，请先执行: pip install modelscope"
    exit 1
fi

download_model() {
    local model_id="$1"
    local local_dir="$2"

    if [ -d "$local_dir" ] && [ "$(ls -A "$local_dir" 2>/dev/null)" ]; then
        log_warn "已存在，跳过: $local_dir"
        return 0
    fi

    log_info "下载 $model_id -> $local_dir"
    mkdir -p "$local_dir"
    modelscope download --model "$model_id" --local_dir "$local_dir"
}

echo "============================================"
echo " FunASR C++ Runtime 模型下载"
echo " 目标目录: ${MODEL_ZOO}"
echo "============================================"
echo ""

# ==================== ASR 语音识别 (ONNX) ====================
log_info "===== ASR 语音识别模型 ====="

download_model "damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx" \
    "${MODEL_ZOO}/asr/Paraformer-zh/onnx"

download_model "damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404-onnx" \
    "${MODEL_ZOO}/asr/Paraformer-zh-contextual/onnx"

download_model "iic/SenseVoiceSmall-onnx" \
    "${MODEL_ZOO}/asr/SenseVoiceSmall/onnx"

download_model "damo/speech_paraformer_asr_nat-zh-cn-8k-common-vocab8358-tensorflow1-onnx" \
    "${MODEL_ZOO}/asr/Paraformer-zh-8k/onnx"

download_model "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx" \
    "${MODEL_ZOO}/asr/Paraformer-zh-streaming/onnx"

# ==================== VAD 语音端点检测 (ONNX) ====================
log_info "===== VAD 语音端点检测模型 ====="

download_model "damo/speech_fsmn_vad_zh-cn-16k-common-onnx" \
    "${MODEL_ZOO}/vad/FSMN-VAD/onnx"

download_model "damo/speech_fsmn_vad_zh-cn-8k-common-onnx" \
    "${MODEL_ZOO}/vad/FSMN-VAD-8k/onnx"

# ==================== 标点恢复 (ONNX) ====================
log_info "===== 标点恢复模型 ====="

download_model "damo/punc_ct-transformer_cn-en-common-vocab471067-large-onnx" \
    "${MODEL_ZOO}/punc/CT-Punc/onnx"

download_model "damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx" \
    "${MODEL_ZOO}/punc/CT-Punc-realtime/onnx"

# ==================== 语言模型 ====================
log_info "===== 语言模型 ====="

download_model "damo/speech_ngram_lm_zh-cn-ai-wesp-fst" \
    "${MODEL_ZOO}/lm/Ngram-LM/fst"

download_model "damo/speech_ngram_lm_zh-cn-ai-wesp-fst-token8358" \
    "${MODEL_ZOO}/lm/Ngram-LM-8k/fst"

# ==================== ITN 逆文本正则化 ====================
log_info "===== ITN 逆文本正则化模型 ====="

download_model "thuduj12/fst_itn_zh" \
    "${MODEL_ZOO}/itn/FST-ITN/fst"

# ==================== 说话人模型 CAM++ (PyTorch) ====================
log_info "===== 说话人模型 CAM++ (PyTorch) ====="

download_model "iic/speech_campplus_sv_zh-cn_16k-common" \
    "${MODEL_ZOO}/speaker/CAM++/pytorch"

echo ""
log_info "全部模型下载完成！"
echo ""
echo "目录结构:"
find "${MODEL_ZOO}" -maxdepth 3 -type d | sort | while read -r dir; do
    depth=$(echo "$dir" | sed "s|${MODEL_ZOO}||" | tr -cd '/' | wc -c)
    indent=$(printf '%*s' $((depth * 2)) '')
    echo "  ${indent}$(basename "$dir")/"
done
