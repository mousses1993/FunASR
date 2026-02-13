# FunASR 说话人分离功能使用指南

本文档介绍如何在 FunASR C++ Runtime 中使用说话人分离 (Speaker Diarization) 功能。

## 功能概述

说话人分离是指识别音频中"谁在什么时候说话"的任务。FunASR 实现了基于 CAM++ 模型和谱聚类的说话人分离功能。

### 主要特性

- **CAM++ 说话人嵌入模型**: 轻量级 (7.2M 参数)，提取 192 维说话人嵌入向量
- **谱聚类算法**: 自动识别说话人数量，支持 1-15 个说话人
- **VAD 集成**: 结合语音活动检测，只分析语音段
- **后处理优化**: 自动合并、平滑短片段

---

## 新增文件

### 头文件

| 文件 | 描述 |
|------|------|
| `include/campplus-model.h` | CAM++ 说话人嵌入模型接口 |
| `include/speaker-diarization.h` | 说话人分离模块接口 |

### 源文件

| 文件 | 描述 |
|------|------|
| `src/campplus-model.cpp` | CAM++ 模型 ONNX 推理实现 |
| `src/speaker-diarization.cpp` | 谱聚类和后处理实现 |

### 测试程序

| 文件 | 描述 |
|------|------|
| `bin/funasr-onnx-speaker-diarization.cpp` | 说话人分离测试程序 |

---

## API 参考

### CAM++ 模型 API

```cpp
// 初始化 CAM++ 模型
FUNASR_HANDLE CampplusInit(std::map<std::string, std::string>& model_path, int thread_num);

// 释放模型
void CampplusUninit(FUNASR_HANDLE handle);

// 提取说话人嵌入向量
std::vector<float> CampplusExtractEmbedding(FUNASR_HANDLE handle, 
    const float* features, int num_frames, int feat_dim);
```

### 说话人分离 API

```cpp
// 初始化说话人分离
FUNASR_HANDLE SpeakerDiarizationInit(FUNASR_HANDLE campplus_handle, 
    std::map<std::string, std::string>& config);

// 释放资源
void SpeakerDiarizationUninit(FUNASR_HANDLE handle);

// 执行说话人分离
// 返回 JSON 格式字符串: [[start, end, speaker_id], ...]
const char* SpeakerDiarizationProcess(FUNASR_HANDLE handle, 
    const std::vector<std::tuple<float, float, std::vector<float>>>& vad_segments,
    int sample_rate);

// 释放结果字符串
void SpeakerDiarizationFreeResult(const char* result);
```

### 配置参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `segment_duration` | 1.5 | 分段时长 (秒) |
| `segment_shift` | 0.75 | 分段步长 (秒) |
| `min_num_speakers` | 1 | 最小说话人数 |
| `max_num_speakers` | 15 | 最大说话人数 |
| `merge_threshold` | 0.78 | 基于余弦相似度的合并阈值 |
| `min_segment_duration` | 0.7 | 最小片段时长 (秒) |

---

## 使用示例

### 命令行使用

```bash
# 基本用法
./funasr-onnx-speaker-diarization \
    --speaker-dir /path/to/campplus-model \
    --vad-dir /path/to/vad-model \
    --wav-path /path/to/audio.wav

# 指定说话人数量范围
./funasr-onnx-speaker-diarization \
    --speaker-dir /path/to/campplus-model \
    --vad-dir /path/to/vad-model \
    --wav-path /path/to/meeting.wav \
    --min-speakers 2 \
    --max-speakers 5 \
    --output result.json
```

### 代码示例

```cpp
#include "funasrruntime.h"
#include "campplus-model.h"
#include "speaker-diarization.h"

int main() {
    // 1. 初始化 CAM++ 模型
    std::map<std::string, std::string> model_path;
    model_path["speaker-dir"] = "/path/to/campplus-model";
    
    FUNASR_HANDLE campplus_handle = CampplusInit(model_path, 1);
    if (!campplus_handle) {
        return -1;
    }
    
    // 2. 初始化说话人分离
    std::map<std::string, std::string> config;
    config["min_num_speakers"] = "2";
    config["max_num_speakers"] = "10";
    
    FUNASR_HANDLE diar_handle = SpeakerDiarizationInit(campplus_handle, config);
    if (!diar_handle) {
        CampplusUninit(campplus_handle);
        return -1;
    }
    
    // 3. 准备音频数据 (从 VAD 获取或直接使用)
    std::vector<std::tuple<float, float, std::vector<float>>> vad_segments;
    // ... 添加音频分段 ...
    
    // 4. 执行说话人分离
    const char* result = SpeakerDiarizationProcess(diar_handle, vad_segments, 16000);
    printf("Diarization result: %s\n", result);
    
    // 5. 清理
    SpeakerDiarizationFreeResult(result);
    SpeakerDiarizationUninit(diar_handle);
    CampplusUninit(campplus_handle);
    
    return 0;
}
```

---

## 模型获取

### CAM++ 模型

从 ModelScope 下载:

```bash
# 使用 modelscope 命令行
pip install modelscope
modelscope download --model iic/speech_campplus_sv_zh-cn_16k-common --local_dir ./campplus-model

# 或直接下载
# https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common
```

### 转换为 ONNX 格式

> ⚠️ **重要警告**
>
> **官方目前未提供 CAM++ 的 ONNX 版本模型，也未提供官方的导出工具。**
>
> CAM++ 模型在 FunASR 中**没有实现 `export()` 方法**，以下导出脚本是**非官方的实验性方案**，存在以下风险：
> - 导出的 ONNX 模型可能与原始 PyTorch 模型存在精度差异
> - 某些模型结构可能无法正确导出
> - 没有官方测试和验证
>
> 建议关注官方更新，等待 FunASR 官方提供 ONNX 版本模型或官方导出工具。

### 可选方案：使用第三方 ONNX 模型

有一个第三方社区项目提供了 CAM++ 的 ONNX 版本：

| 项目 | 地址 | 说明 |
|------|------|------|
| **lovemefan/campplus** | https://github.com/lovemefan/campplus | 第三方 CAM++ ONNX 实现 |

**特点**：
- 模型大小约 28MB
- 不需要 PyTorch/torchaudio 依赖
- 直接从内存加载 ONNX 模型

> ⚠️ 注意：这是第三方项目，使用前请自行评估风险和兼容性。

### 实验性导出脚本（仅供参考，不保证正确性）

```python
import torch
from funasr import AutoModel

# 加载模型
model = AutoModel(model="iic/speech_campplus_sv_zh-cn_16k-common")

# 导出 ONNX（实验性，可能存在问题）
dummy_input = torch.randn(1, 100, 80)  # [batch, frames, features]
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

# 建议：导出后务必验证 ONNX 模型与 PyTorch 模型的输出是否一致
```

### 验证 ONNX 导出结果

如果尝试导出 ONNX，建议进行验证：

```python
import torch
import onnxruntime as ort
import numpy as np
from funasr import AutoModel

# 加载 PyTorch 模型
model = AutoModel(model="iic/speech_campplus_sv_zh-cn_16k-common")
model.model.eval()

# 测试输入
test_input = torch.randn(1, 100, 80)

# PyTorch 推理
with torch.no_grad():
    pytorch_output = model.model(test_input).numpy()

# ONNX 推理
ort_session = ort.InferenceSession("campplus.onnx")
onnx_output = ort_session.run(None, {"features": test_input.numpy()})[0]

# 比较输出
diff = np.abs(pytorch_output - onnx_output).max()
print(f"最大差异: {diff}")
# 如果差异过大（如 > 0.01），说明导出可能有问题
```

---

## 性能优化建议

1. **GPU 加速**: 使用 ONNX Runtime GPU 版本可显著提升推理速度
2. **批处理**: 对多个音频段进行批处理嵌入提取
3. **分段策略**: 根据实际场景调整分段参数
4. **聚类优化**: 对于已知说话人数的场景，设置 `min_speakers` 和 `max_speakers` 相同可提升准确性

---

## 常见问题

### Q: 说话人数量识别不准确?

A: 尝试调整以下参数:
- 增大/减小 `merge_threshold` (默认 0.78)
- 手动设置准确的 `min_speakers` 和 `max_speakers`

### Q: 短音频识别效果差?

A: 说话人分离需要足够的语音数据来提取可靠的嵌入向量。建议:
- 音频时长至少 5 秒以上
- 每个说话人至少有 2 秒以上的语音

### Q: 如何与 ASR 结合?

A: 使用 `FunOfflineInferWithSpeaker` API 或自行实现:
1. 先运行 VAD 获取语音段
2. 运行 ASR 获取识别文本
3. 运行说话人分离获取说话人标签
4. 根据时间对齐合并结果

---

## 更新日志

### v1.0.0 (2025-02-13)
- 初始实现
- 支持 CAM++ 嵌入提取
- 支持谱聚类算法
- 支持 VAD 集成
- 添加测试程序

---

*文档生成日期: 2025-02-13*
