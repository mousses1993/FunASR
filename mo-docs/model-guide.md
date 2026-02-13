# FunASR 模型使用指南

本文档介绍如何下载 FunASR 运行所需模型、各功能可用的模型选择，以及如何获取最新模型列表。

---

## 目录

- [1. 模型下载方法](#1-模型下载方法)
  - [1.1 Python 方式下载](#11-python-方式下载)
  - [1.2 命令行方式下载](#12-命令行方式下载)
  - [1.3 直接下载（浏览器）](#13-直接下载浏览器)
  - [1.4 服务端自动下载](#14-服务端自动下载)
- [2. 各功能推荐模型](#2-各功能推荐模型)
  - [2.1 语音识别模型（ASR）](#21-语音识别模型asr)
  - [2.2 语音端点检测模型（VAD）](#22-语音端点检测模型vad)
  - [2.3 标点恢复模型](#23-标点恢复模型)
  - [2.4 说话人分离/验证模型](#24-说话人分离验证模型)
  - [2.5 语言模型](#25-语言模型)
  - [2.6 逆文本正则化模型（ITN）](#26-逆文本正则化模型itn)
  - [2.7 时间戳预测模型](#27-时间戳预测模型)
  - [2.8 情感识别模型](#28-情感识别模型)
  - [2.9 语音唤醒模型](#29-语音唤醒模型)
- [3. 获取最新模型列表](#3-获取最新模型列表)
- [4. 模型格式说明](#4-模型格式说明)
- [5. 常见问题](#5-常见问题)

---

## 1. 模型下载方法

FunASR 模型主要托管在 **ModelScope** 和 **Hugging Face** 两个平台。

### 1.1 Python 方式下载

#### 使用 FunASR AutoModel（推荐）

```python
from funasr import AutoModel

# 自动从 ModelScope 下载模型
model = AutoModel(model="paraformer-zh")  # 会自动下载

# 指定模型缓存目录
model = AutoModel(
    model="paraformer-zh",
    hub="ms"  # ms = ModelScope, hf = Hugging Face
)
```

#### 使用 ModelScope SDK

```python
# 安装 modelscope
# pip install modelscope

from modelscope.hub.snapshot_download import snapshot_download

# 下载模型到指定目录
model_dir = snapshot_download(
    "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    cache_dir="./models"
)
print(f"模型已下载到: {model_dir}")
```

### 1.2 命令行方式下载

#### ModelScope CLI

```bash
# 安装 modelscope
pip install modelscope

# 下载模型到指定目录
modelscope download --model iic/speech_campplus_sv_zh-cn_16k-common --local_dir ./campplus-model

# 下载 ONNX 格式模型
modelscope download --model iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx --local_dir ./asr-model
```

#### Hugging Face CLI

```bash
# 安装 huggingface_hub
pip install huggingface_hub

# 下载模型
huggingface-cli download --local-dir ./model-dir FunAudioLLM/SenseVoiceSmall
```

### 1.3 直接下载（浏览器）

直接访问模型页面下载：

| 平台 | 地址 |
|------|------|
| ModelScope | https://modelscope.cn/models?page=1&tasks=auto-speech-recognition |
| Hugging Face | https://huggingface.co/FunASR |

### 1.4 服务端自动下载

FunASR 服务端支持启动时自动下载模型：

```bash
# 离线文件转写服务
cd FunASR/runtime
nohup bash run_server.sh \
  --download-model-dir /workspace/models \
  --vad-dir damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
  --model-dir damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx \
  --punc-dir damo/punc_ct-transformer_cn-en-common-vocab471067-large-onnx > log.txt 2>&1 &
```

服务会自动从 ModelScope 下载 `--xxx-dir` 参数指定的模型到 `--download-model-dir` 目录。

---

## 2. 各功能推荐模型

### 2.1 语音识别模型（ASR）

#### 非实时语音识别（离线）

| 模型名称 | ModelScope ID | 参数量 | 特点 |
|---------|--------------|-------|------|
| **Fun-ASR-Nano** | `FunAudioLLM/Fun-ASR-Nano-2512` | 800M | 最新大模型，支持31种语言，支持方言、歌词识别 |
| **SenseVoiceSmall** | `iic/SenseVoiceSmall` | 330M | 多功能语音理解（ASR+LID+SER+AED） |
| **Paraformer-zh** | `damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch` | 220M | 中文语音识别，带时间戳输出 |
| **SeACo-Paraformer** | `iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` | 220M | 支持热词功能的语音识别 |
| **Paraformer-zh-spk** | `damo/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn` | 220M | 分角色语音识别 |
| **Paraformer-en** | `damo/speech_paraformer-large-vad-punc_asr_nat-en-16k-common-vocab10020` | 220M | 英文语音识别 |
| **Whisper-large-v3** | `iic/Whisper-large-v3` | 1550M | 多语言语音识别 |
| **Whisper-large-v3-turbo** | `iic/Whisper-large-v3-turbo` | 809M | 多语言语音识别，更快 |

**ONNX 格式模型（用于 C++ Runtime 部署）：**

| 模型用途 | ONNX 模型 ID |
|---------|-------------|
| 标准识别 | `damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx` |
| 时间戳输出 | `damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx` |
| NN热词 | `damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404-onnx` |
| SenseVoice | `iic/SenseVoiceSmall-onnx` |
| 8k采样率 | `damo/speech_paraformer_asr_nat-zh-cn-8k-common-vocab8358-tensorflow1-onnx` |

#### 实时语音识别（流式）

| 模型名称 | ModelScope ID | 参数量 | 特点 |
|---------|--------------|-------|------|
| **Paraformer-zh-streaming** | `damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online` | 220M | 中文实时语音识别 |
| **Paraformer-zh-streaming-small** | `iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online` | 220M | 轻量版实时识别 |

**ONNX 格式（用于 C++ Runtime）：**

```
damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx
```

### 2.2 语音端点检测模型（VAD）

| 模型名称 | ModelScope ID | 参数量 | 用途 |
|---------|--------------|-------|------|
| **FSMN-VAD** | `damo/speech_fsmn_vad_zh-cn-16k-common-pytorch` | 0.4M | 语音端点检测 |

**ONNX 格式：**

| 采样率 | ONNX 模型 ID |
|-------|-------------|
| 16k | `damo/speech_fsmn_vad_zh-cn-16k-common-onnx` |
| 8k | `damo/speech_fsmn_vad_zh-cn-8k-common-onnx` |

### 2.3 标点恢复模型

| 模型名称 | ModelScope ID | 参数量 | 特点 |
|---------|--------------|-------|------|
| **CT-Punc** | `damo/punc_ct-transformer_cn-en-common-vocab471067-large` | 290M | 中英文标点恢复 |

**ONNX 格式：**

| 模型用途 | ONNX 模型 ID |
|---------|-------------|
| 标准标点恢复 | `damo/punc_ct-transformer_cn-en-common-vocab471067-large-onnx` |
| 实时模式 | `damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx` |

### 2.4 说话人分离/验证模型

| 模型名称 | ModelScope ID | 参数量 | 特点 |
|---------|--------------|-------|------|
| **CAM++** | `iic/speech_campplus_sv_zh-cn_16k-common` | 7.2M | 说话人确认/分割，192维嵌入向量 |

**使用示例：**

```bash
# 下载 CAM++ 模型
modelscope download --model iic/speech_campplus_sv_zh-cn_16k-common --local_dir ./campplus-model
```

### 2.5 语言模型

| 模型名称 | ModelScope ID | 用途 |
|---------|--------------|------|
| **Ngram LM (zh-cn)** | `damo/speech_ngram_lm_zh-cn-ai-wesp-fst` | 中文 Ngram 语言模型 |
| **Ngram LM (8k)** | `damo/speech_ngram_lm_zh-cn-ai-wesp-fst-token8358` | 8k采样率版本 |

### 2.6 逆文本正则化模型（ITN）

| 模型名称 | ModelScope ID | 用途 |
|---------|--------------|------|
| **FST-ITN** | `thuduj12/fst_itn_zh` | 中文逆文本正则化（数字、日期等） |

### 2.7 时间戳预测模型

| 模型名称 | ModelScope ID | 参数量 | 用途 |
|---------|--------------|-------|------|
| **FA-zh** | `damo/speech_timestamp_prediction-v1-16k-offline` | 38M | 字级别时间戳预测 |

### 2.8 情感识别模型

| 模型名称 | ModelScope ID | 参数量 | 情感类别 |
|---------|--------------|-------|---------|
| **emotion2vec+large** | `iic/emotion2vec_plus_large` | 300M | 生气、开心、中立、难过 |
| **emotion2vec+base** | `iic/emotion2vec_plus_base` | - | 同上 |
| **emotion2vec+seed** | `iic/emotion2vec_plus_seed` | - | 同上 |

### 2.9 语音唤醒模型

| 模型名称 | ModelScope ID | 用途 |
|---------|--------------|------|
| **FSMN-KWS** | `iic/speech_sanm_kws_phone-xiaoyun-commands-online` | 语音唤醒（在线） |
| **SANM-KWS** | `iic/speech_sanm_kws_phone-xiaoyun-commands-offline` | 语音唤醒（离线） |

---

## 3. 获取最新模型列表

### ModelScope 平台

1. **网页浏览**：
   - 访问 https://modelscope.cn/models?page=1&tasks=auto-speech-recognition
   - 筛选任务类型为"语音识别"等相关任务

2. **按组织筛选**：
   - 阿里达摩 (`damo`): https://modelscope.cn/organization/damo
   - 阿里智能计算研究院 (`iic`): https://modelscope.cn/organization/iic
   - FunAudioLLM: https://modelscope.cn/organization/FunAudioLLM

3. **API 方式**：

```python
from modelscope.hub.api import HubApi

api = HubApi()
# 搜索语音识别模型
models = api.list_models(filter="auto-speech-recognition")
for model in models:
    print(f"{model.name}: {model.id}")
```

### Hugging Face 平台

1. **网页浏览**：
   - FunASR 组织: https://huggingface.co/FunASR
   - FunAudioLLM 组织: https://huggingface.co/FunAudioLLM

2. **API 方式**：

```python
from huggingface_hub import list_models

# 搜索 FunASR 相关模型
models = list_models(author="FunASR")
for model in models:
    print(f"{model.id}")
```

### GitHub 仓库

查看 FunASR 官方模型仓库文档：
- https://github.com/alibaba-damo-academy/FunASR/blob/main/model_zoo/readme_zh.md

---

## 4. 模型格式说明

### PyTorch 格式 vs ONNX 格式

| 格式 | 用途 | 优点 |
|------|------|------|
| **PyTorch** | Python 推理、训练、微调 | 灵活，支持训练 |
| **ONNX** | C++ Runtime 部署 | 高效推理，跨平台 |

### 并非所有 PyTorch 模型都能转为 ONNX

**重要说明**：FunASR 中大部分核心模型已经实现了 `export()` 方法，可以导出为 ONNX 格式。但并非所有模型都支持转换。

#### 支持导出 ONNX 的模型

以下模型类型在 FunASR 中已实现 ONNX 导出功能：

| 模型类型 | 支持状态 | 说明 |
|---------|---------|------|
| Paraformer 系列 | ✅ 支持 | 包括标准版、流式版、热词版等 |
| SenseVoice | ✅ 支持 | 多功能语音理解模型 |
| FSMN-VAD | ✅ 支持 | 语音端点检测 |
| CT-Transformer (Punc) | ✅ 支持 | 标点恢复模型 |
| E-Paraformer | ✅ 支持 | 增强版 Paraformer |
| SANM-KWS | ✅ 支持 | 语音唤醒模型 |
| BICIF-Paraformer | ✅ 支持 | 双向 CIF Paraformer |
| Contextual Paraformer | ✅ 支持 | 上下文感知 Paraformer |
| SeACo-Paraformer | ✅ 支持 | 热词增强 Paraformer |

#### 可能不支持导出的模型

| 模型类型 | 支持状态 | 原因 |
|---------|---------|------|
| 大型多模态模型 (Qwen-Audio) | ❌ 不支持 | 模型结构复杂，依赖多模态组件 |
| Whisper | ⚠️ 部分支持 | 建议直接下载 ONNX 版本 |
| emotion2vec | ⚠️ 有限支持 | 需要手动处理导出 |
| CAM++ (说话人) | ⚠️ 无官方支持 | **官方未提供 export() 方法和 ONNX 版本**，可考虑第三方方案 |

#### 第三方 ONNX 资源

对于官方未提供 ONNX 版本的模型，可以关注第三方社区资源：

| 模型 | 第三方项目 | 说明 |
|------|-----------|------|
| CAM++ | [lovemefan/campplus](https://github.com/lovemefan/campplus) | 提供 ONNX 版本，约 28MB |

> ⚠️ 注意：第三方资源未经官方验证，使用前请自行评估风险。

#### ONNX 转换的限制

PyTorch 模型转换为 ONNX 有以下技术限制：

1. **动态控制流**：包含复杂条件分支的模型可能无法正确导出
2. **自定义算子**：使用了 PyTorch 自定义算子的模型需要额外实现
3. **动态形状**：某些动态形状操作在 ONNX 中支持有限
4. **第三方依赖**：依赖外部库（如 transformers）的模型可能无法直接导出

### 转换 ONNX 格式

#### 方法 1：使用 FunASR AutoModel（推荐）

```python
from funasr import AutoModel

# 加载 PyTorch 模型
model = AutoModel(model="paraformer-zh")

# 导出为 ONNX 格式
res = model.export(quantize=False)  # quantize=True 启用量化
print(f"模型已导出到: {res}")
```

#### 方法 2：命令行导出

```bash
# 基本导出
funasr-export ++model=paraformer-zh ++quantize=false

# 指定导出目录
funasr-export ++model=paraformer-zh ++quantize=true ++output_dir=./exported_models

# 指定 opset 版本
funasr-export ++model=paraformer-zh ++opset_version=14
```

#### 方法 3：手动导出（有风险，仅供参考）

> ⚠️ **警告**：手动导出是非官方方案，可能存在问题。部分模型（如 CAM++）官方未提供 `export()` 方法，手动导出的 ONNX 模型可能与原始模型存在差异。

对于 CAM++ 等需要手动导出的模型（**实验性，不保证正确性**）：

```python
import torch
from funasr import AutoModel

# 加载模型
model = AutoModel(model="iic/speech_campplus_sv_zh-cn_16k-common")

# 创建示例输入
dummy_input = torch.randn(1, 100, 80)  # [batch, frames, features]

# 导出 ONNX
torch.onnx.export(
    model.model,
    dummy_input,
    "campplus.onnx",
    input_names=["features"],
    output_names=["embedding"],
    dynamic_axes={
        "features": {0: "batch", 1: "frames"},
        "embedding": {0: "batch"}
    },
    opset_version=14
)

# 建议：导出后务必验证 ONNX 与 PyTorch 输出是否一致
```

### 导出参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `type` | `onnx` | 导出类型（`onnx` 或 `torchscript`） |
| `quantize` | `False` | 是否启用量化 |
| `opset_version` | `14` | ONNX opset 版本 |
| `calib_num` | `100` | 量化校准样本数 |
| `max_seq_len` | `512` | 最大序列长度 |

### ONNX 模型文件结构

下载或导出的 ONNX 模型目录通常包含：

```
model-dir/
├── model.onnx          # 非量化模型
├── model_quant.onnx    # 量化模型（更小更快）
├── config.yaml         # 模型配置
├── am.mvn              # 均值方差归一化参数
└── tokens.txt          # 词表文件
```

### 建议

1. **优先下载 ONNX 版本**：ModelScope 上大多数模型已提供预导出的 ONNX 版本（模型名带 `-onnx` 后缀）
2. **使用量化模型**：对于生产部署，推荐使用量化版本 (`model_quant.onnx`)，体积更小、推理更快
3. **验证导出结果**：导出后建议用 `funasr-onnx` 库测试验证

---

## 5. 常见问题

### Q: 模型下载速度慢怎么办？

A: 
- ModelScope 国内用户推荐使用 ModelScope 镜像
- 可以设置代理或使用镜像源

```bash
# 使用国内镜像安装 modelscope
pip install modelscope -i https://mirror.sjtu.edu.cn/pypi/web/simple
```

### Q: 如何查看已下载的模型？

A: 
```python
# 默认缓存位置
# Linux/Mac: ~/.cache/modelscope/hub/
# Windows: C:\Users\<username>\.cache\modelscope\hub\

import os
cache_dir = os.path.expanduser("~/.cache/modelscope/hub")
print(os.listdir(cache_dir))
```

### Q: 如何使用本地已下载的模型？

A:
```python
from funasr import AutoModel

# 直接指定本地路径
model = AutoModel(model="/path/to/local/model-dir")
```

### Q: 量化模型和非量化模型有什么区别？

A:
- **量化模型 (model_quant.onnx)**: 更小的模型体积，更快的推理速度，轻微精度损失
- **非量化模型 (model.onnx)**: 原始精度，适合对精度要求高的场景

### Q: 服务端启动时如何选择模型组合？

A: 根据 `run_server.sh` 参数配置：

```bash
# 离线转写服务推荐配置
bash run_server.sh \
  --vad-dir damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
  --model-dir damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx \
  --punc-dir damo/punc_ct-transformer_cn-en-common-vocab471067-large-onnx \
  --lm-dir damo/speech_ngram_lm_zh-cn-ai-wesp-fst \
  --itn-dir thuduj12/fst_itn_zh
```

---

## 参考链接

- [FunASR GitHub](https://github.com/alibaba-damo-academy/FunASR)
- [FunASR 模型仓库](https://github.com/alibaba-damo-academy/FunASR/blob/main/model_zoo/readme_zh.md)
- [ModelScope 语音识别模型](https://modelscope.cn/models?page=1&tasks=auto-speech-recognition)
- [Hugging Face FunASR](https://huggingface.co/FunASR)
- [FunASR 服务部署文档](../runtime/readme_cn.md)

---

*文档生成日期: 2026-02-13*
