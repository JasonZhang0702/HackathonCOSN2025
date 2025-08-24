import whisper
import numpy as np
import torch
import os
import argparse
import subprocess
from typing import Tuple, Dict, Optional


# 检查ffmpeg是否安装
def check_ffmpeg() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            check = True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


# 加载音频并预处理（Whisper标准流程）
def load_and_preprocess_audio(audio_path: str, sample_rate: int = 16000) -> np.ndarray:
    """加载音频并转换为Whisper所需的格式（16kHz单声道）"""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    # 使用Whisper内置函数加载音频（自动处理采样率和声道）
    audio = whisper.load_audio(audio_path)
    # 确保音频长度至少为模型所需的最小长度
    audio = whisper.pad_or_trim(audio)
    return audio


# 提取核心特征
def extract_whisper_features(
        audio_path: str,
        model_name: str = "medium",
        language: Optional[str] = None,
        output_dir: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    提取Whisper处理过程中的关键音频特征

    返回:
        features: 包含多种特征的字典
            - mel_spectrogram: 梅尔频谱图 (shape: [80, 3000])
            - encoder_hidden_states: 编码器隐藏层特征 (shape: [n_layers, 1500, n_features])
            - logits: 模型输出的原始logits (shape: [1500, vocab_size])
    """
    # 检查依赖
    if not check_ffmpeg():
        raise RuntimeError("未找到ffmpeg，请安装并配置环境变量")

    # 加载模型
    model = whisper.load_model(model_name)
    device = model.device

    # 加载并预处理音频
    audio = load_and_preprocess_audio(audio_path)
    # 转换为张量并移动到模型设备
    audio_tensor = torch.from_numpy(audio).to(device)

    # 提取梅尔频谱图（Whisper的核心输入特征）
    mel = whisper.log_mel_spectrogram(audio_tensor)  # shape: [1, 80, 3000]
    mel_np = mel.squeeze(0).cpu().numpy()  # 去除批次维度并转换为numpy

    # 提取模型中间特征（通过修改forward过程获取）
    with torch.no_grad():  # 禁用梯度计算，节省内存
        # 编码阶段（获取编码器特征）
        encoder_output = model.encoder(mel)  # shape: [1, 1500, n_features]

        # 解码阶段（获取logits）
        decoder_input = torch.tensor([[whisper.tokenizer.sot]]).to(device)  # 起始token
        logits = model.decoder(decoder_input, encoder_output)  # shape: [1, 1, vocab_size]

        # 获取编码器所有层的隐藏状态（如果需要更细粒度的特征）
        encoder_hidden_states = model.encoder.layers_output  # 列表，每个元素是对应层的输出

    # 整理特征
    features = {
        # 梅尔频谱图（原始音频特征）
        "mel_spectrogram": mel_np,
        # 编码器最后一层输出（高级语义特征）
        "encoder_final_hidden": encoder_output.squeeze(0).cpu().numpy(),
        # 所有编码器层的隐藏状态（用于更灵活的特征选择）
        "encoder_hidden_states": np.stack([layer.squeeze(0).cpu().numpy() for layer in encoder_hidden_states]),
        # 模型输出的logits（可用于后续语言模型任务）
        "logits": logits.squeeze(0).cpu().numpy()
    }

    # 保存特征（如果指定了输出目录）
    if output_dir:
        os.makedirs(output_dir, exist_ok = True)
        for name, feat in features.items():
            np.save(os.path.join(output_dir, f"{name}.npy"), feat)
        print(f"特征已保存到: {output_dir}")

    return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "使用Whisper提取音频特征")
    parser.add_argument("--audio_path", default = "D:/pythonProject/lang_psy/hackathon/audio_0.wav")
    parser.add_argument("--model", default = "base", help = "Whisper模型名称（tiny, base, small, medium, large）")
    parser.add_argument("--output", default = "output")

    args = parser.parse_args()

    try:
        features = extract_whisper_features(
            audio_path = args.audio_path,
            model_name = args.model,
            output_dir = args.output
        )

        # 打印特征信息
        print("\n提取的特征信息:")
        for name, feat in features.items():
            print(f"- {name}: 形状 {feat.shape}, 数据类型 {feat.dtype}")

    except Exception as e:
        print(f"提取特征失败: {str(e)}")
