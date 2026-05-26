import copy
import os
import pickle
from typing import Dict, Optional, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy.signal import lfilter

from models import AdaInVC


def inv_mel_matrix(sample_rate: int, n_fft: int, n_mels: int) -> np.array:
    """멜 필터뱅크의 의사역행렬 계산
    
    멜 스펙트로그램을 다시 선형 스펙트로그램으로 변환하는 데 사용된다.
    
    Args:
        sample_rate: 샘플링 레이트
        n_fft: FFT 윈도우 크기
        n_mels: 멜 필터뱅크 개수
        
    Returns:
        멜 필터뱅크의 의사역행렬
    """
    m = librosa.filters.mel(sample_rate, n_fft, n_mels)
    p = np.matmul(m, m.T)
    d = [1.0 / x if np.abs(x) > 1e-8 else x for x in np.sum(p, axis=0)]
    return np.matmul(m.T, np.diag(d))


def normalize(mel: np.array, attr: Dict) -> np.array:
    """멜 스펙트로그램 정규화
    
    Args:
        mel: 멜 스펙트로그램
        attr: 평균과 표준편차를 포함하는 속성 딕셔너리
        
    Returns:
        정규화된 멜 스펙트로그램
    """
    mean, std = attr["mean"], attr["std"]
    mel = (mel - mean) / std
    return mel


def denormalize(mel: np.array, attr: Dict) -> np.array:
    """멜 스펙트로그램 역정규화
    
    Args:
        mel: 정규화된 멜 스펙트로그램
        attr: 평균과 표준편차를 포함하는 속성 딕셔너리
        
    Returns:
        원래 스케일의 멜 스펙트로그램
    """
    mean, std = attr["mean"], attr["std"]
    mel = mel * std + mean
    return mel


def file2mel(
    audio_path: str,
    sample_rate: int,
    preemph: float,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int,
    ref_db: float,
    max_db: float,
    top_db: float,
) -> np.array:
    """오디오 파일을 멜 스펙트로그램으로 변환
    
    Args:
        audio_path: 오디오 파일 경로
        sample_rate: 샘플링 레이트
        preemph: 프리엠퍼시 계수
        n_fft: FFT 윈도우 크기
        hop_length: 홉 길이
        win_length: 윈도우 길이
        n_mels: 멜 필터뱅크 개수
        ref_db: 기준 데시벨 값
        max_db: 최대 데시벨 값
        top_db: 무음 제거 임계값
        
    Returns:
        멜 스펙트로그램
    """
    # 오디오 로드
    wav, _ = librosa.load(audio_path, sr=sample_rate)
    
    # 무음 구간 제거
    wav, _ = librosa.effects.trim(wav, top_db=top_db)
    
    # 프리엠퍼시 적용
    wav = np.append(wav[0], wav[1:] - preemph * wav[:-1])
    
    # 선형 스펙트로그램 계산
    linear = librosa.stft(
        y=wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length
    )
    mag = np.abs(linear)

    # 멜 스펙트로그램 계산
    mel_basis = librosa.filters.mel(sample_rate, n_fft, n_mels)
    mel = np.dot(mel_basis, mag)

    # 데시벨 스케일로 변환하고 정규화
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mel = np.clip((mel - ref_db + max_db) / max_db, 1e-8, 1)
    mel = mel.T.astype(np.float32)

    return mel


def mel2wav(
    mel: np.array,
    sample_rate: int,
    preemph: float,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int,
    ref_db: float,
    max_db: float,
    top_db: float,
) -> np.array:
    """멜 스펙트로그램을 파형으로 변환
    
    Args:
        mel: 멜 스펙트로그램
        sample_rate: 샘플링 레이트
        preemph: 프리엠퍼시 계수
        n_fft: FFT 윈도우 크기
        hop_length: 홉 길이
        win_length: 윈도우 길이
        n_mels: 멜 필터뱅크 개수
        ref_db: 기준 데시벨 값
        max_db: 최대 데시벨 값
        top_db: 무음 제거 임계값
        
    Returns:
        파형 데이터
    """
    # 전치 후 데시벨 스케일 복원
    mel = mel.T
    mel = (np.clip(mel, 0, 1) * max_db) - max_db + ref_db
    mel = np.power(10.0, mel * 0.05)
    
    # 선형 스펙트로그램으로 변환
    inv_mat = inv_mel_matrix(sample_rate, n_fft, n_mels)
    mag = np.dot(inv_mat, mel)
    
    # Griffin-Lim 알고리즘으로 위상 재구성
    wav = griffin_lim(mag, hop_length, win_length, n_fft)
    
    # 역프리엠퍼시 적용
    wav = lfilter([1], [1, -preemph], wav)

    return wav.astype(np.float32)


def griffin_lim(
    spect: np.array,
    hop_length: int,
    win_length: int,
    n_fft: int,
    n_iter: Optional[int] = 100,
) -> np.array:
    """Griffin-Lim 알고리즘으로 위상 재구성
    
    크기 스펙트럼으로부터 위상 정보를 재구성한다.
    
    Args:
        spect: 크기 스펙트럼
        hop_length: 홉 길이
        win_length: 윈도우 길이
        n_fft: FFT 윈도우 크기
        n_iter: 반복 횟수
        
    Returns:
        재구성된 파형
    """
    X_best = copy.deepcopy(spect)
    for _ in range(n_iter):
        X_t = librosa.istft(X_best, hop_length, win_length, window="hann")
        est = librosa.stft(X_t, n_fft, hop_length, win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spect * phase
    X_t = librosa.istft(X_best, hop_length, win_length, window="hann")
    y = np.real(X_t)
    return y


def load_model(model_dir: str) -> Tuple[nn.Module, Dict, Dict, str]:
    """모델과 관련 설정 로드
    
    Args:
        model_dir: 모델 파일 디렉터리
        
    Returns:
        model: 로드된 모델
        config: 모델 설정
        attr: 데이터 속성. 평균과 표준편차
        device: 연산 장치
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    attr_path = os.path.join(model_dir, "attr.pkl")
    cfg_path = os.path.join(model_dir, "config.yaml")
    model_path = os.path.join(model_dir, "model.ckpt")

    # 데이터 속성, 설정, 모델 로드
    attr = pickle.load(open(attr_path, "rb"))
    config = yaml.safe_load(open(cfg_path, "r"))
    model = AdaInVC(config["model"]).to(device)
    model.load_state_dict(torch.load(model_path))

    return model, config, attr, device