# FunASR ONNX Runtime 编译目标说明文档

本文档整理了 FunASR `runtime/onnxruntime` 目录下所有 CMake 编译目标（Targets）的功能和用途。

## 目录

- [库目标 (Library Targets)](#库目标-library-targets)
- [可执行文件目标 (Executable Targets)](#可执行文件目标-executable-targets)
- [第三方依赖库](#第三方依赖库)
- [核心模块说明](#核心模块说明)

---

## 库目标 (Library Targets)

### `funasr` (共享库)

**输出文件**: `libfunasr.so`

**描述**: FunASR 的核心共享库，包含所有语音识别相关的功能模块。

**源文件**: `runtime/onnxruntime/src/` 目录下的所有 `.cpp` 文件

**主要依赖**:
- onnxruntime - ONNX 模型推理引擎
- ffmpeg (avutil, avcodec, avformat, swresample) - 音视频处理
- yaml-cpp - YAML 配置解析
- kaldi-native-fbank - Kaldi Fbank 特征提取
- kaldi-decoder - Kaldi 解码器
- openfst - FST (Finite State Transducer) 操作
- glog - Google 日志库
- gflags - Google 命令行参数库

**包含的核心模块**:
| 模块文件 | 功能描述 |
|---------|---------|
| `paraformer.cpp/h` | Paraformer 非自回归语音识别模型 |
| `paraformer-online.cpp/h` | Paraformer 流式识别模型 |
| `paraformer-torch.cpp/h` | Paraformer GPU (LibTorch) 版本 |
| `fsmn-vad.cpp/h` | FSMN 语音活动检测模型 |
| `fsmn-vad-online.cpp/h` | FSMN 流式 VAD 模型 |
| `ct-transformer.cpp/h` | CT-Transformer 标点预测模型 |
| `ct-transformer-online.cpp/h` | CT-Transformer 流式标点模型 |
| `sensevoice-small.cpp/h` | SenseVoice 小型多语言语音模型 |
| `itn-processor.cpp/h` | 逆文本标准化 (ITN) 处理器 |
| `wfst-decoder.cpp/h` | WFST 解码器（用于热词/LM） |
| `bias-lm.cpp/h` | 偏置语言模型 |
| `funasrruntime.cpp` | FunASR 运行时 API 封装 |

---

## 可执行文件目标 (Executable Targets)

所有可执行文件位于 `runtime/onnxruntime/build/bin/` 目录。

### 离线识别相关

#### 1. `funasr-onnx-offline`

**描述**: 离线文件转写主程序，用于对完整音频文件进行语音识别。

**功能**:
- 支持多种音频格式输入 (wav, pcm, 以及 ffmpeg 支持的格式)
- 支持 VAD 语音活动检测
- 支持标点恢复
- 支持热词功能
- 支持 ITN 逆文本标准化
- 支持 WFST 语言模型解码
- 支持 GPU 加速推理

**典型使用场景**: 对录音文件、长音频、视频音频进行离线转写

**示例命令**:
```bash
./funasr-onnx-offline \
    --model-dir /path/to/paraformer model \
    --vad-dir /path/to/vad-model \
    --punc-dir /path/to/punc-model \
    --wav-path /path/to/audio.wav
```

---

#### 2. `funasr-onnx-offline-vad`

**描述**: 独立的 VAD（语音活动检测）测试程序。

**功能**:
- 检测音频中的语音段
- 输出语音段的起止时间戳

**典型使用场景**: 单独测试 VAD 模型效果，或获取音频中的语音片段信息

---

#### 3. `funasr-onnx-offline-punc`

**描述**: 独立的标点恢复测试程序。

**功能**:
- 对无标点文本添加标点符号

**典型使用场景**: 单独测试标点模型效果

---

#### 4. `funasr-onnx-offline-rtf`

**描述**: 离线识别 RTF (Real-Time Factor) 性能测试程序。

**功能**:
- 测量离线识别的实时率 (RTF)
- 用于性能基准测试

**典型使用场景**: 性能评估、模型对比

---

### 流式识别相关

#### 5. `funasr-onnx-online-vad`

**描述**: 流式 VAD 测试程序。

**功能**:
- 模拟流式场景下的语音活动检测
- 支持分块输入音频

**典型使用场景**: 实时语音检测场景测试

---

#### 6. `funasr-onnx-online-asr`

**描述**: 流式语音识别测试程序。

**功能**:
- 模拟流式语音识别
- 支持分块接收音频数据

**典型使用场景**: 实时语音识别场景测试

---

#### 7. `funasr-onnx-online-punc`

**描述**: 流式标点恢复测试程序。

**功能**:
- 模拟流式场景下的标点预测
- 支持增量式标点添加

**典型使用场景**: 实时语音识别中的标点预测

---

#### 8. `funasr-onnx-online-rtf`

**描述**: 流式识别 RTF 性能测试程序。

**功能**:
- 测量流式识别的实时率 (RTF)
- 用于流式识别性能基准测试

**典型使用场景**: 流式识别性能评估

---

### 2Pass 模式相关

#### 9. `funasr-onnx-2pass`

**描述**: 两阶段识别测试程序，结合流式和离线模型。

**功能**:
- 支持三种模式: `online`、`offline`、`2pass`
- 实时输出流式识别结果
- 句尾使用离线模型进行修正
- 支持 VAD、标点、热词、ITN
- 输出时间戳信息

**典型使用场景**: 
- 实时语音听写
- 需要低延迟响应但又要高精度的场景

**示例命令**:
```bash
./funasr-onnx-2pass \
    --offline-model-dir /path/to/paraformer-offline \
    --online-model-dir /path/to/paraformer-online \
    --vad-dir /path/to/vad-model \
    --punc-dir /path/to/punc-model \
    --asr-mode 2pass \
    --wav-path /path/to/audio.wav
```

---

#### 10. `funasr-onnx-2pass-rtf`

**描述**: 两阶段识别 RTF 性能测试程序。

**功能**:
- 测量 2Pass 模式的实时率 (RTF)
- 用于两阶段识别性能基准测试

**典型使用场景**: 2Pass 模式性能评估

---

### 说话人分离相关

#### 11. `funasr-onnx-speaker-diarization`

**描述**: 说话人分离测试程序，用于识别音频中的不同说话人。

**功能**:
- 基于 CAM++ 模型提取说话人嵌入向量
- 使用谱聚类算法进行说话人聚类
- 支持 VAD 语音分段
- 输出每个说话人的时间区间

**输入**:
- 音频文件 (wav, pcm 或 ffmpeg 支持的格式)
- CAM++ 说话人模型
- VAD 模型 (可选，用于语音分段)

**输出**:
- JSON 格式的说话人时间轴
- 格式: `[[start_time, end_time, speaker_id], ...]`

**典型使用场景**: 
- 会议录音分析
- 多人对话场景
- 客服质检
- 采访音频处理

**示例命令**:
```bash
./funasr-onnx-speaker-diarization \
    --speaker-dir /path/to/campplus-model \
    --vad-dir /path/to/vad-model \
    --wav-path /path/to/meeting.wav \
    --min-speakers 2 \
    --max-speakers 10 \
    --output result.json
```

**输出示例**:
```json
{
  "audio_file": "meeting.wav",
  "duration": 120.5,
  "segments": [
    [0.0, 5.2, 0],
    [5.2, 12.8, 1],
    [12.8, 18.5, 0],
    ...
  ]
}
```

---

## 第三方依赖库

| 依赖库名 | 用途 |
|---------|-----|
| `glog` | Google 日志库，用于日志记录 |
| `gflags` | Google 命令行参数解析库 |
| `openfst` | 有限状态转换器库，用于 WFST 解码和 ITN |
| `yaml-cpp` | YAML 配置文件解析 |
| `kaldi-native-fbank` | Kaldi Fbank 特征提取 |
| `kaldi-decoder` | Kaldi 解码器组件 |
| `json` (nlohmann/json) | JSON 数据处理 |

---

## 核心模块说明

### 语音识别模型

#### Paraformer
- **文件**: `paraformer.cpp/h`, `paraformer-online.cpp/h`
- **论文**: [Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition](https://arxiv.org/pdf/2206.08317.pdf)
- **特点**: 非自回归并行解码，高精度快速识别

#### SenseVoice Small
- **文件**: `sensevoice-small.cpp/h`
- **特点**: 多语言语音识别模型，支持情感识别、语音事件检测

### 语音活动检测 (VAD)

#### FSMN-VAD
- **文件**: `fsmn-vad.cpp/h`, `fsmn-vad-online.cpp/h`
- **论文**: [Deep-FSMN for Large Vocabulary Continuous Speech Recognition](https://arxiv.org/abs/1803.05030)
- **功能**: 检测音频中的语音段，区分语音和静音

### 说话人分离 (Speaker Diarization)

#### CAM++ 说话人嵌入模型
- **文件**: `campplus-model.cpp/h`
- **特点**: 轻量级说话人确认模型，提取 192 维说话人嵌入向量
- **模型来源**: [3D-Speaker](https://github.com/alibaba-damo-academy/3D-Speaker)
- **功能**: 从音频中提取说话人特征嵌入

#### 说话人分离模块
- **文件**: `speaker-diarization.cpp/h`
- **算法**: 谱聚类 (Spectral Clustering) + K-means
- **功能**: 
  - 音频分段与嵌入提取
  - 基于相似度的说话人聚类
  - 后处理优化 (合并、平滑)
- **配置参数**:
  - `min_num_speakers`: 最小说话人数 (默认 1)
  - `max_num_speakers`: 最大说话人数 (默认 15)
  - `merge_threshold`: 合并阈值 (默认 0.78)

### 标点恢复

#### CT-Transformer
- **文件**: `ct-transformer.cpp/h`, `ct-transformer-online.cpp/h`
- **论文**: [CT-Transformer: Controllable time-delay transformer for real-time punctuation prediction and disfluency detection](https://arxiv.org/pdf/2003.01309.pdf)
- **功能**: 为识别结果添加标点符号

### 逆文本标准化 (ITN)

#### ITN Processor
- **文件**: `itn-processor.cpp/h`, `itn-token-parser.cpp/h`
- **功能**: 将口语化数字、日期等转换为书面形式
- **示例**: "一百二十三" → "123"

### WFST 解码器

#### WFST Decoder
- **文件**: `wfst-decoder.cpp/h`
- **功能**: 
  - 支持 N-gram 语言模型
  - 支持热词 (Hotword) 加权
  - 基于 FST 的束搜索解码

### 辅助模块

| 模块 | 文件 | 功能 |
|-----|------|-----|
| 音频处理 | `audio.cpp/h` | 音频加载、重采样、格式转换 |
| 特征提取 | 使用 kaldi-native-fbank | Fbank 特征提取 |
| 词汇表 | `vocab.cpp/h` | 词汇表管理 |
| 分词词典 | `seg_dict.cpp/h` | 中文分词词典 |
| 编码转换 | `encode_converter.cpp/h` | 字符编码转换 |
| UTF-8 处理 | `utf8-string.cpp/h` | UTF-8 字符串处理 |
| 工具函数 | `util.cpp/h` | 通用工具函数 |

---

## 编译选项

在 CMake 配置时可用的选项：

| 选项 | 默认值 | 说明 |
|-----|-------|------|
| `ENABLE_GLOG` | ON | 是否编译 glog 日志库 |
| `ENABLE_FST` | ON | 是否编译 openfst（ITN 需要） |
| `GPU` | OFF | 是否启用 GPU 支持（需要 LibTorch） |

**示例**:
```bash
# 仅 CPU 版本（默认）
cmake -DCMAKE_BUILD_TYPE=release .. \
    -DONNXRUNTIME_DIR=/path/to/onnxruntime \
    -DFFMPEG_DIR=/path/to/ffmpeg

# GPU 版本
cmake -DCMAKE_BUILD_TYPE=release .. \
    -DONNXRUNTIME_DIR=/path/to/onnxruntime \
    -DFFMPEG_DIR=/path/to/ffmpeg \
    -DGPU=ON
```

---

## 相关文档

- [FunASR Runtime 总体说明](../runtime/readme_cn.md)
- [中文离线文件转写服务](../runtime/docs/SDK_advanced_guide_offline_zh.md)
- [中文实时语音听写服务](../runtime/docs/SDK_advanced_guide_online_zh.md)
- [英文离线文件转写服务](../runtime/docs/SDK_advanced_guide_offline_en_zh.md)
- [GPU 版本部署指南](../runtime/docs/SDK_advanced_guide_offline_gpu_zh.md)

---

*文档生成日期: 2025-02-13*
*基于 FunASR 仓库最新版本整理*
