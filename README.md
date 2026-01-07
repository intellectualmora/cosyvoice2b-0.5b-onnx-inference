# ONNX TTS Standalone

基于 ONNX 的独立 TTS 推理管线，适用于 CosyVoice2 - **无需 CosyVoice 源码依赖**。

## 功能特性

- 零样本语音克隆（使用说话人提示音频）
- 流式合成，支持增量音频输出
- 纯 ONNX 推理（支持 CPU/CUDA）
- 交互模式，支持连续 TTS 合成

## 快速开始


### 交互模式

```bash
# 交互模式（默认启用流式输出）
# --prompt-wav 和 --prompt-text 用于声音克隆，提供目标说话人的音频和对应文本
# 启动后在交互界面输入要合成的文本，系统会用克隆的声音朗读
uv run python onnx_tts_standalone.py --interactive \
    --prompt-wav speaker_2961_022.wav \
    --prompt-text "We do not know how Plato would have arranged his own dialects, or whether the thought of arranging any of them, besides the two trilogies, which he has expressly connected, was ever present to his mind."
```

交互示例：
```
> Hello, how are you today?
  [生成音频...]
```


## 所需 ONNX 模型

将以下模型放置在 `models/` 目录下：

| 模型 | 说明 |
|------|------|
| `campplus.onnx` | 说话人嵌入 (192维) |
| `speech_tokenizer.onnx` | 语音分词器 (Whisper mel → tokens) |
| `text_embedding.onnx` | 文本 token 嵌入 |
| `speech_embedding.onnx` | 语音 token 嵌入 (896维，用于 LLM) |
| `llm_embedding.onnx` | LLM 特殊 token (SOS/TaskID) |
| `llm_decoder.onnx` | LLM 输出投影层 |
| `cosyvoice_llm_ar128-cl2048.onnx` | LLM 预填充 (ar128) |
| `cosyvoice_llm_ar1-cl2048.onnx` | LLM 解码 (ar1) |
| `flow_input_embedding.onnx` | Flow token 嵌入 (512维) |
| `flow_spk_embed.onnx` | Flow 说话人投影 (192→80) |
| `flow_encoder_streaming.onnx` | Flow 编码器 |
| `flow_estimator_streaming.onnx` | Flow CFM 估计器 |
| `hift_streaming.onnx` | HiFT 声码器 |

Tokenizer 文件（`tokenizer.json`、`tokenizer_config.json` 等）也需要放在 `models/` 目录下。

## 管线架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        前端 (Frontend)                           │
│  prompt_wav → CAM++ → spk_embedding (192维)                     │
│            → Speech Tokenizer → prompt_speech_tokens            │
│            → Mel Spectrogram → prompt_speech_feat               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        语言模型 (LLM)                            │
│  [SOS, text, TaskID, prompt_speech] → Qwen2 → speech_tokens     │
│  (注意: speaker embedding 不在 LLM 中使用，仅在 Flow 中使用)      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        流匹配 (Flow)                             │
│  speech_tokens → Flow Encoder → μ                               │
│  spk_embedding → Flow Spk Embed → spks (80维)                   │
│  CFM 去噪 (10步) → mel spectrogram                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        声码器 (HiFT)                             │
│  mel → HiFT Vocoder → audio (24kHz)                             │
└─────────────────────────────────────────────────────────────────┘
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--onnx-dir` | ONNX 模型目录 | `models` |
| `--text` | 要合成的文本 | - |
| `--prompt-wav` | 提示音频文件（必需） | - |
| `--prompt-text` | 提示文本 | `""` |
| `--output` | 输出 WAV 文件 | `output.wav` |
| `--interactive` | 交互模式 | `False` |
| `--streaming` | 启用流式合成 | `False` |
| `--no-streaming` | 在交互模式中禁用流式 | `False` |
| `--token-chunk-size` | 每个流式块的 token 数 | `50` |
| `--sampling-top-k` | LLM top-k 采样 | `25` |

## 依赖项

- `numpy` - 数值计算
- `onnxruntime` - ONNX 推理
- `torch` / `torchaudio` - 音频处理、fbank 计算
- `openai-whisper` - Mel 频谱计算
- `transformers` - Qwen tokenizer
- `librosa` - Mel 滤波器组
- `soundfile` - 音频文件读写

## 许可证

本项目仅供研究和教育用途。
