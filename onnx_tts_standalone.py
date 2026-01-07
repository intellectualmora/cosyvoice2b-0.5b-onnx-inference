#!/usr/bin/env python
"""
完全独立的 ONNX TTS Pipeline - 不依赖 CosyVoice 源码

用法:
    python onnx_tts_standalone.py --text "你好世界" --prompt-wav crs_0122.wav --prompt-text "希望你以后能够做的比我还好呦。"
    python onnx_tts_standalone.py --interactive

依赖:
    pip install onnxruntime torch torchaudio tiktoken whisper librosa numpy

ONNX 模型 (放在 models/ 目录):
    - campplus.onnx
    - speech_tokenizer.onnx
    - text_embedding.onnx
    - speech_embedding.onnx
    - llm_embedding.onnx
    - llm_decoder.onnx
    - cosyvoice_llm_ar128-cl2048.onnx
    - cosyvoice_llm_ar1-cl2048.onnx
    - flow_input_embedding.onnx
    - flow_spk_embed.onnx
    - flow_encoder_streaming.onnx
    - flow_estimator_streaming.onnx
    - hift_streaming.onnx
"""

import os
import sys
import argparse
import time
import wave
import numpy as np
import onnxruntime as ort

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import whisper
from transformers import AutoTokenizer
from librosa.filters import mel as librosa_mel_fn

# 尝试导入 soundfile 作为后备音频加载方案
try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


# ============================================================================
# 配置参数
# ============================================================================

SAMPLE_RATE = 24000  # CosyVoice2 输出采样率
SAMPLE_RATE_16K = 16000  # CAM++ 和 Speech Tokenizer 输入采样率

# LLM 配置
NUM_LAYERS = 24
NUM_KV_HEADS = 2
HEAD_DIM = 64
HIDDEN_SIZE = 896
CONTEXT_LENGTH = 2048
ROPE_THETA = 1000000.0
MASK_NEG = -100.0

# Flow 配置
FLOW_TOKEN_LEN = 256
FLOW_MEL_LEN = 512
N_TIMESTEPS = 10
INFERENCE_CFG_RATE = 0.7

# HiFT 配置
HIFT_MEL_CHUNK_LEN = 512
HIFT_CACHE_SIZE = 9
HIFT_NOISE_DIM = 9

# Mel spectrogram 配置 (CosyVoice2)
MEL_N_FFT = 1920
MEL_NUM_MELS = 80
MEL_HOP_SIZE = 480
MEL_WIN_SIZE = 1920
MEL_FMIN = 0
MEL_FMAX = 8000


# ============================================================================
# 工具函数
# ============================================================================

def create_session(onnx_path: str, providers: list = None):
    """创建 ONNX Runtime session"""
    if providers is None:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(onnx_path, sess_options, providers=providers)


def save_wav(audio: np.ndarray, filepath: str, sample_rate: int = SAMPLE_RATE):
    """保存音频为 WAV 文件"""
    if audio.ndim == 2:
        audio = audio.squeeze(0)
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    
    with wave.open(filepath, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())
    
    duration = len(audio) / sample_rate
    print(f"Saved: {filepath} ({duration:.2f}s)")


class StreamingWavWriter:
    """流式 WAV 写入器 - 支持增量写入"""
    
    def __init__(self, filepath: str, sample_rate: int = SAMPLE_RATE):
        self.filepath = filepath
        self.sample_rate = sample_rate
        self.samples_written = 0
        self.file = wave.open(filepath, 'wb')
        self.file.setnchannels(1)
        self.file.setsampwidth(2)
        self.file.setframerate(sample_rate)
    
    def write_chunk(self, audio: np.ndarray):
        """写入一个音频 chunk"""
        if audio.ndim == 2:
            audio = audio.squeeze(0)
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)
        self.file.writeframes(audio_int16.tobytes())
        self.samples_written += len(audio_int16)
    
    def close(self):
        if self.file:
            self.file.close()
            self.file = None
    
    def get_duration(self) -> float:
        return self.samples_written / self.sample_rate
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


def load_audio(filepath: str, target_sr: int) -> torch.Tensor:
    """加载音频文件，支持多种后端"""
    # 优先尝试 soundfile (更稳定，无 torchcodec 依赖)
    if HAS_SOUNDFILE:
        data, sr = sf.read(filepath, dtype='float32')
        if data.ndim == 2:
            data = data.mean(axis=1)  # 转单声道
        waveform = torch.from_numpy(data).unsqueeze(0)  # [1, T]
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        return waveform
    
    # 后备: torchaudio (可能需要 torchcodec)
    waveform, sr = torchaudio.load(filepath)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform


# ============================================================================
# Mel Spectrogram (与 CosyVoice 一致)
# ============================================================================

_mel_basis = {}
_hann_window = {}

def compute_mel_spectrogram_24k(audio: torch.Tensor) -> torch.Tensor:
    """
    计算 Mel spectrogram (用于 Flow 的 prompt_speech_feat)
    
    参数与 CosyVoice2 配置一致:
    - n_fft: 1920
    - num_mels: 80
    - sampling_rate: 24000
    - hop_size: 480
    - win_size: 1920
    - fmin: 0
    - fmax: 8000
    """
    global _mel_basis, _hann_window
    
    device = audio.device
    key = f"{MEL_FMAX}_{device}"
    
    if key not in _mel_basis:
        mel = librosa_mel_fn(sr=SAMPLE_RATE, n_fft=MEL_N_FFT, n_mels=MEL_NUM_MELS, 
                            fmin=MEL_FMIN, fmax=MEL_FMAX)
        _mel_basis[key] = torch.from_numpy(mel).float().to(device)
        _hann_window[str(device)] = torch.hann_window(MEL_WIN_SIZE).to(device)
    
    # Pad
    y = torch.nn.functional.pad(
        audio.unsqueeze(1), 
        (int((MEL_N_FFT - MEL_HOP_SIZE) / 2), int((MEL_N_FFT - MEL_HOP_SIZE) / 2)), 
        mode="reflect"
    )
    y = y.squeeze(1)
    
    # STFT
    spec = torch.view_as_real(
        torch.stft(
            y,
            MEL_N_FFT,
            hop_length=MEL_HOP_SIZE,
            win_length=MEL_WIN_SIZE,
            window=_hann_window[str(device)],
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )
    
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
    spec = torch.matmul(_mel_basis[key], spec)
    spec = torch.log(torch.clamp(spec, min=1e-5))  # dynamic_range_compression
    
    return spec


# ============================================================================
# RoPE Embedding
# ============================================================================

class RopeEmbedding:
    """预计算 RoPE embeddings"""
    
    def __init__(self, head_dim: int = HEAD_DIM, max_length: int = CONTEXT_LENGTH, theta: float = ROPE_THETA):
        freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2)[:head_dim // 2] / head_dim))
        t = np.arange(max_length * 2)
        freqs = np.outer(t, freqs)
        self.cos = np.cos(freqs)[:max_length]
        self.sin = np.sin(freqs)[:max_length]
    
    def get_embedding(self, position_ids: np.ndarray):
        cos = self.cos[position_ids]
        sin = self.sin[position_ids]
        return cos[:, np.newaxis, :, :], sin[:, np.newaxis, :, :]


# ============================================================================
# ONNX Model Wrappers
# ============================================================================

class OnnxCamplus:
    """CAM++ 说话人嵌入 (192-dim)"""
    
    def __init__(self, onnx_path: str):
        print(f"  Loading CAM++: {os.path.basename(onnx_path)}")
        self.session = create_session(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.fixed_len = shape[1] if isinstance(shape[1], int) else None
    
    def __call__(self, fbank: np.ndarray) -> np.ndarray:
        """fbank: [1, T, 80] -> embedding: [1, 192]"""
        if self.fixed_len and fbank.shape[1] != self.fixed_len:
            if fbank.shape[1] < self.fixed_len:
                fbank = np.pad(fbank, ((0, 0), (0, self.fixed_len - fbank.shape[1]), (0, 0)))
            else:
                fbank = fbank[:, :self.fixed_len, :]
        return self.session.run(None, {self.input_name: fbank.astype(np.float32)})[0]


class OnnxSpeechTokenizer:
    """Speech Tokenizer (whisper mel -> speech tokens)"""
    
    def __init__(self, onnx_path: str):
        print(f"  Loading Speech Tokenizer: {os.path.basename(onnx_path)}")
        self.session = create_session(onnx_path)
        shape = self.session.get_inputs()[0].shape
        self.fixed_len = shape[2] if isinstance(shape[2], int) else None
    
    def __call__(self, mel: np.ndarray) -> np.ndarray:
        """mel: [1, 128, T] -> tokens: [1, T//4]"""
        original_len = mel.shape[2]
        
        if self.fixed_len:
            if original_len < self.fixed_len:
                mel = np.pad(mel, ((0, 0), (0, 0), (0, self.fixed_len - original_len)))
            elif original_len > self.fixed_len:
                mel = mel[:, :, :self.fixed_len]
            feats_length = np.array([self.fixed_len], dtype=np.int32)
        else:
            feats_length = np.array([original_len], dtype=np.int32)
        
        outputs = self.session.run(None, {
            'feats': mel.astype(np.float32),
            'feats_length': feats_length
        })
        
        expected_tokens = original_len // 4
        return outputs[0][:, :expected_tokens]


class OnnxEmbedding:
    """通用 Embedding layer (int64 输入)"""
    
    def __init__(self, onnx_path: str, name: str = ""):
        print(f"  Loading {name}: {os.path.basename(onnx_path)}")
        self.session = create_session(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
    
    def __call__(self, token_ids: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: token_ids.astype(np.int64)})[0]


class OnnxLinear:
    """通用 Linear layer (float32 输入)"""
    
    def __init__(self, onnx_path: str, name: str = ""):
        print(f"  Loading {name}: {os.path.basename(onnx_path)}")
        self.session = create_session(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: x.astype(np.float32)})[0]


class OnnxLLMDecoder:
    """LLM Decoder (hidden -> logits)"""
    
    def __init__(self, onnx_path: str):
        print(f"  Loading LLM Decoder: {os.path.basename(onnx_path)}")
        self.session = create_session(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
    
    def __call__(self, hidden: np.ndarray) -> np.ndarray:
        if hidden.ndim == 2:
            hidden = hidden[:, np.newaxis, :]
        return self.session.run(None, {self.input_name: hidden.astype(np.float32)})[0]


class OnnxQwen2:
    """Qwen2 LLM (prefill ar128 + decode ar1)"""
    
    def __init__(self, prefill_path: str, decode_path: str):
        print(f"  Loading LLM Prefill: {os.path.basename(prefill_path)}")
        self.prefill_session = create_session(prefill_path)
        print(f"  Loading LLM Decode: {os.path.basename(decode_path)}")
        self.decode_session = create_session(decode_path)
        
        self.rope = RopeEmbedding()
        self.prefill_seq_len = 128
        self.prefill_past_len = 1920
        self.decode_past_len = 2047
        self.actual_kv_len = 0
    
    def _prepare_attention_mask(self, seq_len: int, past_len: int) -> np.ndarray:
        mask = np.full((1, 1, seq_len, CONTEXT_LENGTH), MASK_NEG, dtype=np.float32)
        valid_start = CONTEXT_LENGTH - past_len - seq_len
        for q in range(seq_len):
            q_abs = valid_start + past_len + q
            mask[0, 0, q, valid_start:q_abs + 1] = 0.0
        return mask
    
    def prefill(self, inputs_embeds: np.ndarray, past_kv: dict = None):
        total_seq_len = inputs_embeds.shape[1]
        
        if total_seq_len <= self.prefill_seq_len:
            return self._prefill_single(inputs_embeds, total_seq_len, 0, None)
        
        kv_cache = None
        hidden_list = []
        processed = 0
        kv_len = 0
        
        while processed < total_seq_len:
            chunk_size = min(self.prefill_seq_len, total_seq_len - processed)
            chunk_xs = inputs_embeds[:, processed:processed + chunk_size, :]
            hidden, kv_cache = self._prefill_single(chunk_xs, chunk_size, kv_len, kv_cache)
            hidden_list.append(hidden)
            kv_len += chunk_size
            processed += chunk_size
        
        self.actual_kv_len = kv_len
        return np.concatenate(hidden_list, axis=1), kv_cache
    
    def _prefill_single(self, inputs_embeds: np.ndarray, seq_len: int, past_kv_len: int, existing_kv: dict):
        if seq_len < self.prefill_seq_len:
            pad_len = self.prefill_seq_len - seq_len
            inputs_embeds = np.pad(inputs_embeds, ((0, 0), (pad_len, 0), (0, 0)))
        
        position_ids = np.zeros((1, self.prefill_seq_len), dtype=np.int64)
        for i in range(seq_len):
            position_ids[0, self.prefill_seq_len - seq_len + i] = past_kv_len + i
        
        cos, sin = self.rope.get_embedding(position_ids)
        attention_mask = self._prepare_attention_mask(self.prefill_seq_len, past_kv_len)
        
        if seq_len < self.prefill_seq_len:
            attention_mask[:, :, :self.prefill_seq_len - seq_len, :] = MASK_NEG
        
        feed_dict = {
            'inputs_embeds': inputs_embeds.astype(np.float32),
            'attention_mask': attention_mask,
            'position_ids_cos': cos.astype(np.float32),
            'position_ids_sin': sin.astype(np.float32),
        }
        
        for i in range(NUM_LAYERS):
            if existing_kv is not None:
                key_cache = existing_kv[f'past_key_{i}']
                value_cache = existing_kv[f'past_value_{i}']
                kv_len = key_cache.shape[3]
                if kv_len < self.prefill_past_len:
                    pad_len = self.prefill_past_len - kv_len
                    key_cache = np.pad(key_cache, ((0, 0), (0, 0), (0, 0), (pad_len, 0)))
                    value_cache = np.pad(value_cache, ((0, 0), (0, 0), (pad_len, 0), (0, 0)))
                feed_dict[f'past_key_{i}_in'] = key_cache
                feed_dict[f'past_value_{i}_in'] = value_cache
            else:
                feed_dict[f'past_key_{i}_in'] = np.zeros((1, NUM_KV_HEADS, HEAD_DIM, self.prefill_past_len), dtype=np.float32)
                feed_dict[f'past_value_{i}_in'] = np.zeros((1, NUM_KV_HEADS, self.prefill_past_len, HEAD_DIM), dtype=np.float32)
        
        outputs = self.prefill_session.run(None, feed_dict)
        hidden = outputs[0]
        if seq_len < self.prefill_seq_len:
            hidden = hidden[:, -seq_len:, :]
        
        new_kv = {}
        for i in range(NUM_LAYERS):
            key_out = outputs[1 + i * 2]
            value_out = outputs[2 + i * 2]
            if seq_len < self.prefill_seq_len:
                key_out = key_out[:, :, :, -seq_len:]
                value_out = value_out[:, :, -seq_len:, :]
            if existing_kv is not None and past_kv_len > 0:
                old_key = existing_kv[f'past_key_{i}'][:, :, :, -past_kv_len:]
                old_value = existing_kv[f'past_value_{i}'][:, :, -past_kv_len:, :]
                key_out = np.concatenate([old_key, key_out], axis=3)
                value_out = np.concatenate([old_value, value_out], axis=2)
            new_kv[f'past_key_{i}'] = key_out
            new_kv[f'past_value_{i}'] = value_out
        
        return hidden, new_kv
    
    def decode_step(self, inputs_embeds: np.ndarray, past_kv: dict, position: int):
        position_ids = np.array([[position]])
        cos, sin = self.rope.get_embedding(position_ids)
        
        attention_mask = np.full((1, 1, 1, CONTEXT_LENGTH), MASK_NEG, dtype=np.float32)
        valid_start = CONTEXT_LENGTH - position - 1
        attention_mask[:, :, :, valid_start:] = 0.0
        
        feed_dict = {
            'inputs_embeds': inputs_embeds.astype(np.float32),
            'attention_mask': attention_mask,
            'position_ids_cos': cos.astype(np.float32),
            'position_ids_sin': sin.astype(np.float32),
        }
        
        cache_len = past_kv['past_key_0'].shape[3]
        for i in range(NUM_LAYERS):
            key_cache = past_kv[f'past_key_{i}']
            value_cache = past_kv[f'past_value_{i}']
            if cache_len < self.decode_past_len:
                pad_len = self.decode_past_len - cache_len
                key_cache = np.pad(key_cache, ((0, 0), (0, 0), (0, 0), (pad_len, 0)))
                value_cache = np.pad(value_cache, ((0, 0), (0, 0), (pad_len, 0), (0, 0)))
            elif cache_len > self.decode_past_len:
                key_cache = key_cache[:, :, :, -self.decode_past_len:]
                value_cache = value_cache[:, :, -self.decode_past_len:, :]
            feed_dict[f'past_key_{i}_in'] = key_cache
            feed_dict[f'past_value_{i}_in'] = value_cache
        
        outputs = self.decode_session.run(None, feed_dict)
        hidden = outputs[0]
        
        new_kv = {}
        for i in range(NUM_LAYERS):
            key_out = outputs[1 + i * 2]
            value_out = outputs[2 + i * 2]
            old_key = past_kv[f'past_key_{i}']
            old_value = past_kv[f'past_value_{i}']
            if old_key.shape[3] >= self.decode_past_len:
                old_key = old_key[:, :, :, 1:]
                old_value = old_value[:, :, 1:, :]
            new_kv[f'past_key_{i}'] = np.concatenate([old_key, key_out], axis=3)
            new_kv[f'past_value_{i}'] = np.concatenate([old_value, value_out], axis=2)
        
        self.actual_kv_len = min(self.actual_kv_len + 1, CONTEXT_LENGTH - 1)
        return hidden, new_kv


class OnnxFlowEncoder:
    """Flow Encoder (token_embeds -> mel)"""
    
    def __init__(self, onnx_path: str, fixed_token_len: int = FLOW_TOKEN_LEN):
        print(f"  Loading Flow Encoder: {os.path.basename(onnx_path)}")
        self.session = create_session(onnx_path)
        self.fixed_token_len = fixed_token_len
        self.fixed_mel_len = fixed_token_len * 2
    
    def __call__(self, token_embeds: np.ndarray, token_len: int) -> tuple:
        seq_len = token_embeds.shape[1]
        if seq_len < self.fixed_token_len:
            token_embeds = np.pad(token_embeds, ((0, 0), (0, self.fixed_token_len - seq_len), (0, 0)))
        elif seq_len > self.fixed_token_len:
            token_embeds = token_embeds[:, :self.fixed_token_len, :]
        
        outputs = self.session.run(None, {
            'token_embeds': token_embeds.astype(np.float32),
            'token_len': np.array([self.fixed_token_len], dtype=np.int64)
        })
        
        actual_mel_len = min(seq_len * 2, self.fixed_mel_len)
        return outputs[0][:, :actual_mel_len, :], actual_mel_len


class OnnxFlowEstimator:
    """Flow Estimator (CFM denoising)"""
    
    def __init__(self, onnx_path: str, fixed_mel_len: int = FLOW_MEL_LEN):
        print(f"  Loading Flow Estimator: {os.path.basename(onnx_path)}")
        self.session = create_session(onnx_path)
        self.fixed_mel_len = fixed_mel_len
    
    def __call__(self, x, mask, mu, t, spks, cond) -> np.ndarray:
        mel_len = x.shape[2]
        if mel_len < self.fixed_mel_len:
            pad_len = self.fixed_mel_len - mel_len
            x = np.pad(x, ((0, 0), (0, 0), (0, pad_len)))
            mask = np.pad(mask, ((0, 0), (0, 0), (0, pad_len)))
            mu = np.pad(mu, ((0, 0), (0, 0), (0, pad_len)))
            cond = np.pad(cond, ((0, 0), (0, 0), (0, pad_len)))
        elif mel_len > self.fixed_mel_len:
            x = x[:, :, :self.fixed_mel_len]
            mask = mask[:, :, :self.fixed_mel_len]
            mu = mu[:, :, :self.fixed_mel_len]
            cond = cond[:, :, :self.fixed_mel_len]
        
        outputs = self.session.run(None, {
            'x': x.astype(np.float32),
            'mask': mask.astype(np.float32),
            'mu': mu.astype(np.float32),
            't': t.astype(np.float32),
            'spks': spks.astype(np.float32),
            'cond': cond.astype(np.float32)
        })
        
        return outputs[0][:, :, :mel_len] if mel_len < self.fixed_mel_len else outputs[0]


class OnnxHiFT:
    """HiFT Vocoder (mel -> audio)"""
    
    def __init__(self, onnx_path: str):
        print(f"  Loading HiFT: {os.path.basename(onnx_path)}")
        self.session = create_session(onnx_path)
        self.mel_chunk_len = HIFT_MEL_CHUNK_LEN
        self.cache_size = HIFT_CACHE_SIZE
        self.noise_dim = HIFT_NOISE_DIM
        self.audio_len = None
        
        for inp in self.session.get_inputs():
            if inp.name == 'mel' and isinstance(inp.shape[2], int):
                self.mel_chunk_len = inp.shape[2]
            elif inp.name == 'cache_in' and isinstance(inp.shape[1], int):
                self.cache_size = inp.shape[1]
            elif inp.name == 'noise' and isinstance(inp.shape[2], int):
                self.noise_dim = inp.shape[2]
        
        for out in self.session.get_outputs():
            if out.name == 'audio' and isinstance(out.shape[1], int):
                self.audio_len = out.shape[1]
    
    def __call__(self, mel: np.ndarray) -> np.ndarray:
        mel_len = mel.shape[2]
        cache = np.zeros((1, self.cache_size), dtype=np.float32)
        all_audio = []
        
        start = 0
        while start < mel_len:
            end = min(start + self.mel_chunk_len, mel_len)
            mel_chunk = mel[:, :, start:end]
            actual_len = end - start
            
            if actual_len < self.mel_chunk_len:
                mel_chunk = np.pad(mel_chunk, ((0, 0), (0, 0), (0, self.mel_chunk_len - actual_len)))
            
            noise = np.random.randn(1, self.mel_chunk_len, self.noise_dim).astype(np.float32)
            outputs = self.session.run(None, {
                'mel': mel_chunk.astype(np.float32),
                'cache_in': cache,
                'noise': noise
            })
            audio_chunk, cache = outputs[0], outputs[1]
            
            if actual_len < self.mel_chunk_len and self.audio_len:
                ratio = actual_len / self.mel_chunk_len
                valid_len = int(audio_chunk.shape[1] * ratio)
                audio_chunk = audio_chunk[:, :valid_len]
            
            all_audio.append(audio_chunk)
            start = end
        
        return np.concatenate(all_audio, axis=1) if all_audio else np.zeros((1, 0), dtype=np.float32)



# ============================================================================
# 完整 TTS Pipeline
# ============================================================================

class OnnxTTSPipeline:
    """完全独立的 ONNX TTS Pipeline"""
    
    def __init__(self, onnx_dir: str):
        print("=" * 60)
        print("Loading ONNX TTS Pipeline (Standalone)")
        print("=" * 60)
        
        self.onnx_dir = onnx_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Frontend
        self.campplus = OnnxCamplus(os.path.join(onnx_dir, 'campplus.onnx'))
        self.speech_tokenizer = OnnxSpeechTokenizer(os.path.join(onnx_dir, 'speech_tokenizer.onnx'))
        
        # 2. LLM Embeddings
        self.text_embedding = OnnxEmbedding(os.path.join(onnx_dir, 'text_embedding.onnx'), 'Text Embedding')
        self.speech_embedding = OnnxEmbedding(os.path.join(onnx_dir, 'speech_embedding.onnx'), 'Speech Embedding (896-dim)')
        self.llm_embedding = OnnxEmbedding(os.path.join(onnx_dir, 'llm_embedding.onnx'), 'LLM Embedding')
        
        # 3. LLM
        self.llm_decoder = OnnxLLMDecoder(os.path.join(onnx_dir, 'llm_decoder.onnx'))
        self.llm = OnnxQwen2(
            os.path.join(onnx_dir, 'cosyvoice_llm_ar128-cl2048.onnx'),
            os.path.join(onnx_dir, 'cosyvoice_llm_ar1-cl2048.onnx')
        )
        
        # 4. Flow
        self.flow_input_embedding = OnnxEmbedding(
            os.path.join(onnx_dir, 'flow_input_embedding.onnx'), 'Flow Input Embedding (512-dim)')
        self.flow_spk_embed = OnnxLinear(
            os.path.join(onnx_dir, 'flow_spk_embed.onnx'), 'Flow Spk Embed (192->80)')
        self.flow_encoder = OnnxFlowEncoder(os.path.join(onnx_dir, 'flow_encoder_streaming.onnx'))
        self.flow_estimator = OnnxFlowEstimator(os.path.join(onnx_dir, 'flow_estimator_streaming.onnx'))
        
        # 5. HiFT
        self.hift = OnnxHiFT(os.path.join(onnx_dir, 'hift_streaming.onnx'))
        
        # 6. Tokenizer (Qwen tokenizer from models folder)
        self.tokenizer = AutoTokenizer.from_pretrained(onnx_dir, trust_remote_code=True)
        # Add special tokens for CosyVoice2
        special_tokens = {
            'eos_token': '<|endoftext|>',
            'pad_token': '<|endoftext|>',
            'additional_special_tokens': [
                '<|im_start|>', '<|im_end|>', '<|endofprompt|>',
                '[breath]', '<strong>', '</strong>', '[noise]',
                '[laughter]', '[cough]', '[clucking]', '[accent]',
                '[quick_breath]', "<laughter>", "</laughter>",
                "[hissing]", "[sigh]", "[vocalized-noise]",
                "[lipsmack]", "[mn]"
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        
        print("=" * 60)
        print("✓ Pipeline loaded!")
        print("=" * 60)
    
    def tokenize_text(self, text: str) -> np.ndarray:
        """文本 tokenize (使用 Qwen tokenizer)"""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return np.array([tokens], dtype=np.int64)
    
    def extract_prompt_features(self, prompt_wav: str):
        """从 prompt 音频提取所有特征"""
        # 1. 加载音频
        audio_16k = load_audio(prompt_wav, SAMPLE_RATE_16K)
        audio_24k = load_audio(prompt_wav, SAMPLE_RATE)
        
        # 2. CAM++ 说话人嵌入 (fbank -> 192-dim)
        fbank = kaldi.fbank(audio_16k, num_mel_bins=80, dither=0, sample_frequency=SAMPLE_RATE_16K)
        fbank = fbank - fbank.mean(dim=0, keepdim=True)
        fbank_np = fbank.unsqueeze(0).numpy()  # [1, T, 80]
        spk_embedding = self.campplus(fbank_np)  # [1, 192]
        
        # 3. Speech tokens (whisper mel -> tokens)
        mel_128 = whisper.log_mel_spectrogram(audio_16k, n_mels=128)  # [1, 128, T]
        mel_128_np = mel_128.numpy()
        speech_tokens = self.speech_tokenizer(mel_128_np)  # [1, T//4]
        
        # 4. Speech feat (24k mel for Flow condition)
        speech_feat = compute_mel_spectrogram_24k(audio_24k)  # [1, 80, T]
        speech_feat = speech_feat.squeeze(0).transpose(0, 1).numpy()  # [T, 80]
        speech_feat = speech_feat[np.newaxis, :, :]  # [1, T, 80]
        
        # 5. 对齐 speech_feat 和 speech_tokens (token_mel_ratio = 2)
        token_len = speech_tokens.shape[1]
        feat_len = speech_feat.shape[1]
        aligned_len = min(feat_len // 2, token_len)
        speech_tokens = speech_tokens[:, :aligned_len]
        speech_feat = speech_feat[:, :aligned_len * 2, :]
        
        return {
            'spk_embedding': spk_embedding,
            'speech_tokens': speech_tokens,
            'speech_feat': speech_feat,
        }
    
    def llm_inference(self, text_tokens: np.ndarray, prompt_text_tokens: np.ndarray,
                      prompt_speech_tokens: np.ndarray, spk_embedding: np.ndarray,
                      sampling_top_k: int = 25, max_tokens: int = 2048) -> np.ndarray:
        """LLM 推理生成 speech tokens"""
        
        # 1. 合并 prompt_text 和 text
        if prompt_text_tokens.shape[1] > 0:
            full_text_tokens = np.concatenate([prompt_text_tokens, text_tokens], axis=1)
        else:
            full_text_tokens = text_tokens
        
        # 2. Text embedding
        text_embeds = self.text_embedding(full_text_tokens)  # [1, seq_len, 896]
        
        # 3. SOS 和 Task ID embedding
        sos_emb = self.llm_embedding(np.array([[0]]))  # [1, 1, 896]
        task_id_emb = self.llm_embedding(np.array([[1]]))  # [1, 1, 896]
        
        # 4. Prompt speech embedding
        if prompt_speech_tokens.shape[1] > 0:
            prompt_speech_emb = self.speech_embedding(prompt_speech_tokens)  # [1, T, 896]
        else:
            prompt_speech_emb = np.zeros((1, 0, HIDDEN_SIZE), dtype=np.float32)
        
        # 5. 构建 LLM 输入: [sos, text, task_id, prompt_speech]
        lm_input = np.concatenate([sos_emb, text_embeds, task_id_emb, prompt_speech_emb], axis=1)
        
        # 6. Prefill
        hidden, kv_cache = self.llm.prefill(lm_input)
        
        # 7. Decode
        out_tokens = []
        speech_token_size = 6561
        eos_token = speech_token_size
        
        for i in range(max_tokens):
            logits = self.llm_decoder(hidden[:, -1:, :]).squeeze(1)  # [1, vocab]
            
            # Top-k sampling
            top_k = min(sampling_top_k, logits.shape[-1])
            top_indices = np.argpartition(logits[0], -top_k)[-top_k:]
            top_logits = logits[0, top_indices]
            probs = np.exp(top_logits - np.max(top_logits))
            probs = probs / probs.sum()
            idx = np.random.choice(len(top_indices), p=probs)
            token = top_indices[idx]
            
            if token >= eos_token:
                break
            
            out_tokens.append(token)
            
            # 下一步
            token_emb = self.speech_embedding(np.array([[token]]))
            position = self.llm.actual_kv_len
            hidden, kv_cache = self.llm.decode_step(token_emb, kv_cache, position)
        
        return np.array([out_tokens], dtype=np.int64)
    
    def flow_inference(self, speech_tokens: np.ndarray, prompt_speech_tokens: np.ndarray,
                       prompt_speech_feat: np.ndarray, spk_embedding: np.ndarray) -> np.ndarray:
        """Flow 推理: speech tokens -> mel (支持分块处理长序列)"""
        
        # 1. 合并 tokens
        full_tokens = np.concatenate([prompt_speech_tokens, speech_tokens], axis=1)
        total_token_len = full_tokens.shape[1]
        
        # 2. Token embedding (Flow 的 512-dim embedding)
        token_embeds = self.flow_input_embedding(full_tokens)  # [1, seq_len, 512]
        
        # 3. Speaker embedding projection (192 -> 80)
        spks = self.flow_spk_embed(spk_embedding)  # [1, 80]
        
        # 4. 分块 Encoder
        encoder_out = self._overlap_chunk_encode(token_embeds)  # [1, mel_len, 80]
        total_mel_len = encoder_out.shape[1]
        
        # 5. 构建 condition
        prompt_mel_len = prompt_speech_feat.shape[1] if prompt_speech_feat is not None else 0
        cond = np.zeros((1, 80, total_mel_len), dtype=np.float32)
        if prompt_speech_feat is not None and prompt_mel_len > 0:
            # 截断或填充 prompt_speech_feat 到 total_mel_len
            copy_len = min(prompt_mel_len, total_mel_len)
            cond[:, :, :copy_len] = prompt_speech_feat[:, :copy_len, :].transpose(0, 2, 1)
        
        # 6. 分块 CFM denoising
        mu = encoder_out.transpose(0, 2, 1)  # [1, 80, mel_len]
        mask = np.ones((1, 1, total_mel_len), dtype=np.float32)
        
        feat = self._overlap_chunk_cfm(mu, mask, spks, cond)
        
        # 7. 只返回生成的部分 (去掉 prompt 对应的 mel)
        prompt_token_len = prompt_speech_tokens.shape[1]
        output_start = prompt_token_len * 2  # token_mel_ratio = 2
        mel = feat[:, :, output_start:]
        return mel
    
    def _overlap_chunk_encode(self, token_embeds: np.ndarray) -> np.ndarray:
        """带重叠的分块 encoder 处理"""
        total_len = token_embeds.shape[1]
        
        if total_len <= FLOW_TOKEN_LEN:
            encoder_out, mel_len = self.flow_encoder(token_embeds, total_len)
            return encoder_out[:, :total_len * 2, :]
        
        # 分块参数
        chunk_size = FLOW_TOKEN_LEN
        overlap = 32  # token 重叠
        hop = chunk_size - overlap
        mel_overlap = overlap * 2
        
        chunks = []
        positions = []
        
        start = 0
        while start < total_len:
            end = min(start + chunk_size, total_len)
            chunk = token_embeds[:, start:end, :]
            chunk_len = end - start
            
            h_chunk, _ = self.flow_encoder(chunk, chunk_len)
            h_chunk = h_chunk[:, :chunk_len * 2, :]
            
            chunks.append(h_chunk)
            positions.append(start * 2)
            
            if end >= total_len:
                break
            start += hop
        
        # 合并 chunks (crossfade)
        total_mel_len = total_len * 2
        h = np.zeros((1, total_mel_len, 80), dtype=np.float32)
        
        fade_in = np.linspace(0, 1, mel_overlap, dtype=np.float32)
        fade_out = np.linspace(1, 0, mel_overlap, dtype=np.float32)
        
        for i, (chunk, pos) in enumerate(zip(chunks, positions)):
            chunk_mel_len = chunk.shape[1]
            end_pos = pos + chunk_mel_len
            
            if i == 0:
                h[:, pos:end_pos, :] = chunk
            else:
                overlap_start = pos
                overlap_end = min(pos + mel_overlap, end_pos)
                actual_overlap = overlap_end - overlap_start
                
                if actual_overlap > 0:
                    fade_in_slice = fade_in[:actual_overlap].reshape(1, actual_overlap, 1)
                    fade_out_slice = fade_out[:actual_overlap].reshape(1, actual_overlap, 1)
                    h[:, overlap_start:overlap_end, :] = (
                        h[:, overlap_start:overlap_end, :] * fade_out_slice +
                        chunk[:, :actual_overlap, :] * fade_in_slice
                    )
                
                if overlap_end < end_pos:
                    h[:, overlap_end:end_pos, :] = chunk[:, actual_overlap:, :]
        
        return h
    
    def _overlap_chunk_cfm(self, mu: np.ndarray, mask: np.ndarray, 
                           spks: np.ndarray, cond: np.ndarray) -> np.ndarray:
        """带重叠的分块 CFM 推理"""
        mel_len = mu.shape[2]
        
        if mel_len <= FLOW_MEL_LEN:
            return self._cfm_inference_single(mu, mask, spks, cond)
        
        # 分块参数
        chunk_size = FLOW_MEL_LEN
        overlap = 64
        hop = chunk_size - overlap
        
        chunks = []
        positions = []
        
        start = 0
        while start < mel_len:
            end = min(start + chunk_size, mel_len)
            
            mu_chunk = mu[:, :, start:end]
            mask_chunk = mask[:, :, start:end]
            cond_chunk = cond[:, :, start:end]
            
            feat_chunk = self._cfm_inference_single(mu_chunk, mask_chunk, spks, cond_chunk)
            
            chunks.append(feat_chunk)
            positions.append(start)
            
            if end >= mel_len:
                break
            start += hop
        
        # 合并 chunks (crossfade)
        feat = np.zeros((1, 80, mel_len), dtype=np.float32)
        
        fade_in = np.linspace(0, 1, overlap, dtype=np.float32)
        fade_out = np.linspace(1, 0, overlap, dtype=np.float32)
        
        for i, (chunk, pos) in enumerate(zip(chunks, positions)):
            chunk_len = chunk.shape[2]
            end_pos = pos + chunk_len
            
            if i == 0:
                feat[:, :, pos:end_pos] = chunk
            else:
                overlap_start = pos
                overlap_end = min(pos + overlap, end_pos)
                actual_overlap = overlap_end - overlap_start
                
                if actual_overlap > 0:
                    fade_in_slice = fade_in[:actual_overlap].reshape(1, 1, actual_overlap)
                    fade_out_slice = fade_out[:actual_overlap].reshape(1, 1, actual_overlap)
                    feat[:, :, overlap_start:overlap_end] = (
                        feat[:, :, overlap_start:overlap_end] * fade_out_slice +
                        chunk[:, :, :actual_overlap] * fade_in_slice
                    )
                
                if overlap_end < end_pos:
                    feat[:, :, overlap_end:end_pos] = chunk[:, :, actual_overlap:]
        
        return feat
    
    def _cfm_inference_single(self, mu: np.ndarray, mask: np.ndarray,
                              spks: np.ndarray, cond: np.ndarray) -> np.ndarray:
        """单个 chunk 的 CFM 推理"""
        mel_len = mu.shape[2]
        
        z = np.random.randn(1, 80, mel_len).astype(np.float32)
        t_span = np.linspace(0, 1, N_TIMESTEPS + 1)
        t_span = 1 - np.cos(t_span * 0.5 * np.pi)
        
        x = z
        for step in range(1, len(t_span)):
            t = t_span[step - 1]
            dt = t_span[step] - t_span[step - 1]
            
            # CFG: batch=2
            x_in = np.concatenate([x, x], axis=0)
            mask_in = np.concatenate([mask, mask], axis=0)
            mu_in = np.concatenate([mu, np.zeros_like(mu)], axis=0)
            t_in = np.array([t, t], dtype=np.float32)
            spks_in = np.concatenate([spks, np.zeros_like(spks)], axis=0)
            cond_in = np.concatenate([cond, np.zeros_like(cond)], axis=0)
            
            dphi_dt = self.flow_estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
            
            dphi_dt_cond = dphi_dt[:1]
            dphi_dt_uncond = dphi_dt[1:]
            dphi_dt = (1.0 + INFERENCE_CFG_RATE) * dphi_dt_cond - INFERENCE_CFG_RATE * dphi_dt_uncond
            
            x = x + dt * dphi_dt
        
        return x
    
    def synthesize(self, text: str, prompt_wav: str, prompt_text: str = "",
                   sampling_top_k: int = 25) -> np.ndarray:
        """完整的 TTS 合成"""
        print(f"\nSynthesizing: {text}")
        
        t_start = time.time()
        
        # 1. 提取 prompt 特征
        print("  [1/5] Extracting prompt features...")
        t0 = time.time()
        prompt_features = self.extract_prompt_features(prompt_wav)
        print(f"        Done ({(time.time()-t0)*1000:.0f}ms)")
        print(f"        spk_embedding: {prompt_features['spk_embedding'].shape}")
        print(f"        speech_tokens: {prompt_features['speech_tokens'].shape}")
        print(f"        speech_feat: {prompt_features['speech_feat'].shape}")
        
        # 2. Tokenize 文本
        print("  [2/5] Tokenizing text...")
        text_tokens = self.tokenize_text(text)
        prompt_text_tokens = self.tokenize_text(prompt_text) if prompt_text else np.zeros((1, 0), dtype=np.int64)
        print(f"        text_tokens: {text_tokens.shape}, prompt_text_tokens: {prompt_text_tokens.shape}")
        
        # 3. LLM 生成 speech tokens
        print("  [3/5] LLM inference...")
        t0 = time.time()
        speech_tokens = self.llm_inference(
            text_tokens, prompt_text_tokens,
            prompt_features['speech_tokens'],
            prompt_features['spk_embedding'],
            sampling_top_k=sampling_top_k
        )
        print(f"        Generated {speech_tokens.shape[1]} speech tokens ({(time.time()-t0)*1000:.0f}ms)")
        
        # 4. Flow 生成 mel
        print("  [4/5] Flow inference...")
        t0 = time.time()
        mel = self.flow_inference(
            speech_tokens,
            prompt_features['speech_tokens'],
            prompt_features['speech_feat'],
            prompt_features['spk_embedding']
        )
        print(f"        Generated mel: {mel.shape} ({(time.time()-t0)*1000:.0f}ms)")
        
        # 5. HiFT 生成音频
        print("  [5/5] HiFT vocoder...")
        t0 = time.time()
        audio = self.hift(mel)
        print(f"        Generated audio: {audio.shape} ({(time.time()-t0)*1000:.0f}ms)")
        
        total_time = time.time() - t_start
        audio_duration = audio.shape[1] / SAMPLE_RATE
        rtf = total_time / audio_duration if audio_duration > 0 else 0
        print(f"\n  Total: {total_time:.2f}s, Audio: {audio_duration:.2f}s, RTF: {rtf:.3f}")
        
        return audio
    
    def synthesize_streaming(self, text: str, prompt_wav: str, prompt_text: str = "",
                             output_path: str = None, sampling_top_k: int = 25,
                             token_chunk_size: int = 50):
        """
        流式 TTS 合成 - LLM 每生成一批 tokens 就立即做 Flow + HiFT 并输出
        
        Args:
            text: 要合成的文本
            prompt_wav: prompt 音频文件
            prompt_text: prompt 文本
            output_path: 输出 WAV 文件路径 (如果提供则增量写入)
            sampling_top_k: LLM top-k 采样
            token_chunk_size: 每多少个 token 输出一次音频
        
        Yields:
            audio_chunk: np.ndarray [1, samples] - 音频片段
        """
        print(f"\nStreaming synthesis: {text}")
        
        t_start = time.time()
        first_chunk_time = None
        
        # 1. 提取 prompt 特征
        print("  [1] Extracting prompt features...")
        prompt_features = self.extract_prompt_features(prompt_wav)
        
        # 2. Tokenize 文本
        text_tokens = self.tokenize_text(text)
        prompt_text_tokens = self.tokenize_text(prompt_text) if prompt_text else np.zeros((1, 0), dtype=np.int64)
        
        # 3. 准备 LLM
        if prompt_text_tokens.shape[1] > 0:
            full_text_tokens = np.concatenate([prompt_text_tokens, text_tokens], axis=1)
        else:
            full_text_tokens = text_tokens
        
        text_embeds = self.text_embedding(full_text_tokens)
        sos_emb = self.llm_embedding(np.array([[0]]))
        task_id_emb = self.llm_embedding(np.array([[1]]))
        
        prompt_speech_tokens = prompt_features['speech_tokens']
        if prompt_speech_tokens.shape[1] > 0:
            prompt_speech_emb = self.speech_embedding(prompt_speech_tokens)
        else:
            prompt_speech_emb = np.zeros((1, 0, HIDDEN_SIZE), dtype=np.float32)
        
        lm_input = np.concatenate([sos_emb, text_embeds, task_id_emb, prompt_speech_emb], axis=1)
        
        # 4. LLM Prefill
        print("  [2] LLM prefill...")
        hidden, kv_cache = self.llm.prefill(lm_input)
        
        # 5. 流式 LLM decode + Flow + HiFT
        print("  [3] Streaming decode...")
        
        wav_writer = None
        if output_path:
            wav_writer = StreamingWavWriter(output_path)
        
        speech_token_size = 6561
        eos_token = speech_token_size
        
        all_tokens = []
        pending_tokens = []
        chunk_idx = 0
        total_audio_samples = 0
        
        try:
            for i in range(2048):  # max tokens
                logits = self.llm_decoder(hidden[:, -1:, :]).squeeze(1)
                
                # Top-k sampling
                top_k = min(sampling_top_k, logits.shape[-1])
                top_indices = np.argpartition(logits[0], -top_k)[-top_k:]
                top_logits = logits[0, top_indices]
                probs = np.exp(top_logits - np.max(top_logits))
                probs = probs / probs.sum()
                idx = np.random.choice(len(top_indices), p=probs)
                token = top_indices[idx]
                
                if token >= eos_token:
                    break
                
                all_tokens.append(token)
                pending_tokens.append(token)
                
                # 每 token_chunk_size 个 token 输出一次
                if len(pending_tokens) >= token_chunk_size:
                    chunk_idx += 1
                    
                    # Flow + HiFT
                    chunk_tokens = np.array([all_tokens], dtype=np.int64)
                    mel = self.flow_inference(
                        chunk_tokens,
                        prompt_features['speech_tokens'],
                        prompt_features['speech_feat'],
                        prompt_features['spk_embedding']
                    )
                    audio_chunk = self.hift(mel)
                    
                    # 计算增量 (去掉之前已输出的部分)
                    new_samples = audio_chunk.shape[1] - total_audio_samples
                    if new_samples > 0:
                        incremental_audio = audio_chunk[:, total_audio_samples:]
                        total_audio_samples = audio_chunk.shape[1]
                        
                        if first_chunk_time is None:
                            first_chunk_time = time.time() - t_start
                            print(f"      First chunk: {first_chunk_time*1000:.0f}ms")
                        
                        chunk_duration = new_samples / SAMPLE_RATE
                        print(f"      Chunk {chunk_idx}: +{new_samples} samples ({chunk_duration:.2f}s), "
                              f"total: {total_audio_samples/SAMPLE_RATE:.2f}s, tokens: {len(all_tokens)}")
                        
                        if wav_writer:
                            wav_writer.write_chunk(incremental_audio)
                        
                        yield incremental_audio
                    
                    pending_tokens = []
                
                # 下一步 LLM
                token_emb = self.speech_embedding(np.array([[token]]))
                position = self.llm.actual_kv_len
                hidden, kv_cache = self.llm.decode_step(token_emb, kv_cache, position)
            
            # 处理剩余的 tokens
            if len(all_tokens) > 0:
                chunk_tokens = np.array([all_tokens], dtype=np.int64)
                mel = self.flow_inference(
                    chunk_tokens,
                    prompt_features['speech_tokens'],
                    prompt_features['speech_feat'],
                    prompt_features['spk_embedding']
                )
                audio_chunk = self.hift(mel)
                
                new_samples = audio_chunk.shape[1] - total_audio_samples
                if new_samples > 0:
                    incremental_audio = audio_chunk[:, total_audio_samples:]
                    total_audio_samples = audio_chunk.shape[1]
                    
                    chunk_duration = new_samples / SAMPLE_RATE
                    print(f"      Final chunk: +{new_samples} samples ({chunk_duration:.2f}s), "
                          f"total: {total_audio_samples/SAMPLE_RATE:.2f}s")
                    
                    if wav_writer:
                        wav_writer.write_chunk(incremental_audio)
                    
                    yield incremental_audio
        
        finally:
            if wav_writer:
                wav_writer.close()
        
        total_time = time.time() - t_start
        audio_duration = total_audio_samples / SAMPLE_RATE
        rtf = total_time / audio_duration if audio_duration > 0 else 0
        print(f"\n  Total: {total_time:.2f}s, Audio: {audio_duration:.2f}s, RTF: {rtf:.3f}")
        print(f"  Generated {len(all_tokens)} speech tokens")


# ============================================================================
# Interactive Mode
# ============================================================================

def interactive_mode(pipeline: OnnxTTSPipeline, prompt_wav: str, prompt_text: str, output_dir: str,
                     streaming: bool = True, token_chunk_size: int = 50):
    """交互式 TTS (支持流式输出)"""
    print("\n" + "=" * 60)
    print("Interactive TTS Mode" + (" (Streaming)" if streaming else ""))
    print("=" * 60)
    print(f"Prompt: {prompt_text[:50]}..." if len(prompt_text) > 50 else f"Prompt: {prompt_text}")
    print(f"Output: {output_dir}")
    if streaming:
        print(f"Token chunk size: {token_chunk_size}")
    print("\nEnter text to synthesize (or 'quit' to exit):")
    print("-" * 60)
    
    session_id = 0
    
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
        
        if not text:
            continue
        if text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        session_id += 1
        output_path = os.path.join(output_dir, f"output_{session_id:03d}.wav")
        
        try:
            if streaming:
                # 流式合成 - 增量写入 WAV
                for audio_chunk in pipeline.synthesize_streaming(
                    text, prompt_wav, prompt_text,
                    output_path=output_path,
                    token_chunk_size=token_chunk_size
                ):
                    pass  # 音频已在 generator 内部写入文件
                print(f"  Saved: {output_path}")
            else:
                # 非流式合成
                audio = pipeline.synthesize(text, prompt_wav, prompt_text)
                save_wav(audio, output_path)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Standalone ONNX TTS Pipeline (No CosyVoice dependency)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 单次合成
  python onnx_tts_standalone.py --text "你好世界" --prompt-wav crs_0122.wav --prompt-text "希望你以后能够做的比我还好呦。"
  
  # 流式合成
  python onnx_tts_standalone.py --text "你好世界" --prompt-wav crs_0122.wav --streaming
  
  # 交互模式 (默认流式)
  python onnx_tts_standalone.py --interactive --prompt-wav crs_0122.wav --prompt-text "希望你以后能够做的比我还好呦。"
        """
    )
    parser.add_argument('--onnx-dir', type=str, default='models',
                        help='Directory containing ONNX models')
    parser.add_argument('--text', type=str, default=None,
                        help='Text to synthesize')
    parser.add_argument('--prompt-wav', type=str, required=True,
                        help='Prompt audio file')
    parser.add_argument('--prompt-text', type=str, default='',
                        help='Prompt text')
    parser.add_argument('--output', type=str, default='output.wav',
                        help='Output WAV file')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory for interactive mode')
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive mode')
    parser.add_argument('--streaming', action='store_true',
                        help='Enable streaming synthesis')
    parser.add_argument('--no-streaming', action='store_true',
                        help='Disable streaming in interactive mode')
    parser.add_argument('--token-chunk-size', type=int, default=50,
                        help='Token chunk size for streaming (default: 50)')
    parser.add_argument('--sampling-top-k', type=int, default=25,
                        help='Top-k sampling for LLM')
    
    args = parser.parse_args()
    
    # 检查 ONNX 目录
    if not os.path.exists(args.onnx_dir):
        print(f"Error: ONNX directory not found: {args.onnx_dir}")
        return 1
    
    # 检查 prompt 文件
    if not os.path.exists(args.prompt_wav):
        print(f"Error: Prompt audio not found: {args.prompt_wav}")
        return 1
    
    # 加载 pipeline
    try:
        pipeline = OnnxTTSPipeline(args.onnx_dir)
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 交互模式
    if args.interactive:
        streaming = not args.no_streaming  # 默认开启流式
        interactive_mode(pipeline, args.prompt_wav, args.prompt_text, args.output_dir,
                        streaming=streaming, token_chunk_size=args.token_chunk_size)
        return 0
    
    # 单次合成
    if args.text:
        try:
            if args.streaming:
                # 流式合成
                for audio_chunk in pipeline.synthesize_streaming(
                    args.text, args.prompt_wav, args.prompt_text,
                    output_path=args.output,
                    sampling_top_k=args.sampling_top_k,
                    token_chunk_size=args.token_chunk_size
                ):
                    pass
                print(f"Saved: {args.output}")
            else:
                # 非流式合成
                audio = pipeline.synthesize(args.text, args.prompt_wav, args.prompt_text, 
                                            sampling_top_k=args.sampling_top_k)
                save_wav(audio, args.output)
            return 0
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    parser.print_help()
    return 0


if __name__ == '__main__':
    exit(main())
