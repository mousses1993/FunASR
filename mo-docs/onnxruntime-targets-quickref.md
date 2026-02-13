# FunASR ONNX Runtime 编译目标快速参考

## 编译产物一览表

### 核心库

| Target | 输出文件 | 说明 |
|--------|---------|------|
| `funasr` | `libfunasr.so` | FunASR 核心共享库 |

### 可执行文件

| Target | 类型 | 功能描述 | 使用场景 |
|--------|-----|---------|---------|
| `funasr-onnx-offline` | 离线ASR | 完整离线文件转写 | 长音频/视频转写 |
| `funasr-onnx-offline-vad` | 离线VAD | VAD 语音检测 | 测试 VAD 模型 |
| `funasr-onnx-offline-punc` | 离线Punc | 标点恢复 | 测试标点模型 |
| `funasr-onnx-offline-rtf` | 离线RTF | 性能测试 | 离线 RTF 基准 |
| `funasr-onnx-online-vad` | 流式VAD | 流式 VAD | 实时语音检测 |
| `funasr-onnx-online-asr` | 流式ASR | 流式识别 | 实时语音识别 |
| `funasr-onnx-online-punc` | 流式Punc | 流式标点 | 实时标点预测 |
| `funasr-onnx-online-rtf` | 流式RTF | 性能测试 | 流式 RTF 基准 |
| `funasr-onnx-2pass` | 2Pass | 两阶段识别 | 实时听写服务 |
| `funasr-onnx-2pass-rtf` | 2Pass RTF | 性能测试 | 2Pass RTF 基准 |

---

## 命令行参数速查

### 通用参数

| 参数 | 说明 | 示例 |
|-----|------|------|
| `--model-dir` | ASR 模型目录 | `--model-dir /models/paraformer` |
| `--quantize` | 是否使用量化模型 | `--quantize true` |
| `--vad-dir` | VAD 模型目录 | `--vad-dir /models/fsmn-vad` |
| `--punc-dir` | 标点模型目录 | `--punc-dir /models/ct-transformer` |
| `--itn-dir` | ITN 模型目录 | `--itn-dir /models/itn` |
| `--lm-dir` | 语言模型目录 | `--lm-dir /models/lm` |
| `--wav-path` | 输入音频路径 | `--wav-path test.wav` |
| `--audio-fs` | 采样率 | `--audio-fs 16000` |
| `--hotword` | 热词文件 | `--hotword hotwords.txt` |

### 2Pass 专用参数

| 参数 | 说明 | 可选值 |
|-----|------|--------|
| `--offline-model-dir` | 离线模型目录 | - |
| `--online-model-dir` | 流式模型目录 | - |
| `--asr-mode` | 识别模式 | `offline`, `online`, `2pass` |

### GPU 参数

| 参数 | 说明 | 示例 |
|-----|------|------|
| `--use-gpu` | 使用 GPU | `--use-gpu` |
| `--batch-size` | 批处理大小 | `--batch-size 4` |

---

## 快速使用示例

### 1. 基础离线识别
```bash
./funasr-onnx-offline \
    --model-dir /path/to/paraformer \
    --wav-path audio.wav
```

### 2. 完整离线识别（VAD + Punc + ITN）
```bash
./funasr-onnx-offline \
    --model-dir /path/to/paraformer \
    --vad-dir /path/to/fsmn-vad \
    --punc-dir /path/to/ct-transformer \
    --itn-dir /path/to/itn \
    --wav-path audio.wav
```

### 3. 2Pass 实时听写
```bash
./funasr-onnx-2pass \
    --offline-model-dir /path/to/paraformer-offline \
    --online-model-dir /path/to/paraformer-online \
    --vad-dir /path/to/fsmn-vad \
    --punc-dir /path/to/ct-transformer \
    --asr-mode 2pass \
    --wav-path audio.wav
```

### 4. 带热词的识别
```bash
# 热词文件格式 (每行: 热词 权重)
echo "阿里巴巴 20" > hotwords.txt
echo "通义实验室 15" >> hotwords.txt

./funasr-onnx-offline \
    --model-dir /path/to/paraformer \
    --hotword hotwords.txt \
    --wav-path audio.wav
```

---

## 核心源文件索引

### 模型实现

| 模块 | 文件 | 功能 |
|-----|------|-----|
| Paraformer | `paraformer.cpp/h` | 离线 ASR |
| Paraformer Online | `paraformer-online.cpp/h` | 流式 ASR |
| Paraformer Torch | `paraformer-torch.cpp/h` | GPU ASR |
| FSMN-VAD | `fsmn-vad.cpp/h` | 离线 VAD |
| FSMN-VAD Online | `fsmn-vad-online.cpp/h` | 流式 VAD |
| CT-Transformer | `ct-transformer.cpp/h` | 离线标点 |
| CT-Transformer Online | `ct-transformer-online.cpp/h` | 流式标点 |
| SenseVoice | `sensevoice-small.cpp/h` | 多语言 ASR |

### 功能模块

| 模块 | 文件 | 功能 |
|-----|------|-----|
| ITN | `itn-processor.cpp/h` | 逆文本标准化 |
| WFST | `wfst-decoder.cpp/h` | WFST 解码 |
| Bias LM | `bias-lm.cpp/h` | 偏置语言模型 |
| Audio | `audio.cpp/h` | 音频处理 |
| Runtime API | `funasrruntime.cpp` | API 封装 |

---

*详细说明请参考: [onnxruntime-build-targets.md](./onnxruntime-build-targets.md)*
