import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import numpy as np
from typing import Tuple, Optional, List


def duration_to_samples(duration_sec: float, sample_rate: int = 16000) -> int:
    """
    수정: 초 단위 길이를 waveform sample 개수로 변환한다.
    예: 1.65초, 16kHz -> 26400 samples
    """
    return int(round(duration_sec * sample_rate))


def duration_to_frames(duration_sec: float, sample_rate: int = 16000, hop_length: int = 256) -> int:
    """
    수정: 초 단위 길이를 mel/STFT frame 개수로 변환한다.
    예: 1.65초, 16kHz, hop=256 -> 약 103 frames
    """
    return int(round(duration_sec * sample_rate / hop_length))


def match_waveform_length(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
    """
    수정: waveform 길이를 target_length에 맞춘다.
    mel_to_waveform 또는 Griffin-Lim 복원 후 원본 길이와 달라지는 문제를 방지한다.

    Args:
        waveform: 입력 waveform [..., T]
        target_length: 목표 길이

    Returns:
        길이가 target_length로 맞춰진 waveform
    """
    current_length = waveform.shape[-1]

    if current_length > target_length:
        return waveform[..., :target_length]

    if current_length < target_length:
        pad_len = target_length - current_length
        return torch.nn.functional.pad(waveform, (0, pad_len))

    return waveform


class MelSpectrogramConverter:
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256, n_mels=80):
        """
        멜 스펙트로그램 변환기
        
        Args:
            sample_rate: 오디오 샘플링 레이트
            n_fft: FFT 크기
            hop_length: 프레임 이동 간격
            n_mels: 멜 필터뱅크 개수
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        
        # 역변환을 위한 근사 변환. 정확한 재구성을 위해서는 외부 vocoder가 필요함
        self.inverse_mel_transform = T.InverseMelScale(
            n_stft=n_fft // 2 + 1,
            n_mels=n_mels,
            sample_rate=sample_rate
        )
        
        self.griffin_lim = T.GriffinLim(
            n_fft=n_fft,
            hop_length=hop_length
        )

    def to(self, device: str):
        """
        수정: converter 내부 torchaudio transform들을 지정한 device로 이동한다.
        GPU 환경에서 waveform과 transform의 device mismatch를 방지한다.
        """
        self.mel_transform = self.mel_transform.to(device)
        self.inverse_mel_transform = self.inverse_mel_transform.to(device)
        self.griffin_lim = self.griffin_lim.to(device)
        return self

    def seconds_to_samples(self, duration_sec: float) -> int:
        """
        수정: 현재 converter 설정 기준으로 초 단위를 sample 개수로 변환한다.
        """
        return duration_to_samples(duration_sec, self.sample_rate)

    def seconds_to_frames(self, duration_sec: float) -> int:
        """
        수정: 현재 converter 설정 기준으로 초 단위를 mel frame 개수로 변환한다.
        """
        return duration_to_frames(duration_sec, self.sample_rate, self.hop_length)
    
    def waveform_to_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        파형을 멜 스펙트로그램으로 변환
        
        Args:
            waveform: 입력 파형 [1, T]
            
        Returns:
            멜 스펙트로그램 [1, n_mels, T']
        """
        # 수정: 입력이 [T] 형태로 들어오면 [1, T]로 변환
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # 로그 멜 스펙트로그램 변환 적용
        mel_spec = self.mel_transform(waveform)

        # 로그 스케일로 변환
        log_mel_spec = torch.log10(torch.clamp(mel_spec, min=1e-5))
        return log_mel_spec
    
    def mel_to_waveform(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        멜 스펙트로그램을 다시 파형으로 변환. 근사 재구성
        
        Args:
            mel_spec: 멜 스펙트로그램 [1, n_mels, T]
            
        Returns:
            재구성된 파형 [1, T']
        """
        # 수정: 입력이 [B, 1, F, T]로 들어오면 batch 첫 번째 샘플만 변환
        # 현재 Griffin-Lim 기반 복원은 단일 waveform 복원 기준으로 사용한다.
        if mel_spec.dim() == 4:
            mel_spec = mel_spec.squeeze(1)
            if mel_spec.shape[0] != 1:
                mel_spec = mel_spec[0:1]

        # 로그 스케일에서 선형 스케일로 변환
        linear_mel_spec = torch.pow(10, mel_spec)

        # 선형 스펙트로그램으로 변환
        spec = self.inverse_mel_transform(linear_mel_spec)

        # Griffin-Lim 알고리즘으로 파형 재구성
        waveform = self.griffin_lim(spec)

        # 수정: GriffinLim 출력이 [T]인 경우 [1, T]로 맞춤
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        return waveform
    
    def apply_weighted_constraint(self, perturbation: torch.Tensor, 
                                 epsilon1: float = 0.1, 
                                 epsilon2: float = 0.05, 
                                 epsilon3: float = 0.08) -> torch.Tensor:
        """
        교란에 가중치 기반 제약 적용
        
        Args:
            perturbation: 멜 스펙트로그램 교란
            epsilon1: 저주파 제약값
            epsilon2: 중주파 제약값
            epsilon3: 고주파 제약값
            
        Returns:
            가중치 기반 제약이 적용된 교란
        """
        # 수정: 기존 구현은 [1, F, T] 형태만 가정했다.
        # 이제 [1, F, T]와 [B, 1, F, T]를 모두 지원한다.
        if perturbation.dim() == 3:
            # 멜 스펙트로그램의 주파수 차원 가져오기
            _, freq_dim, _ = perturbation.shape
            freq_dim_index = 1
            cat_dim = 1
        elif perturbation.dim() == 4:
            # 수정: train_predictive.py, vsmask.py에서는 [B, 1, F, T] 형태를 사용한다.
            _, _, freq_dim, _ = perturbation.shape
            freq_dim_index = 2
            cat_dim = 2
        else:
            raise ValueError(
                f"perturbation은 [1, F, T] 또는 [B, 1, F, T] 형태여야 합니다. "
                f"현재 shape: {tuple(perturbation.shape)}"
            )
        
        # 주파수 범위 정의
        low_freq_end = int(freq_dim * 0.3)  # 저주파 범위. 약 1.6kHz 이하에 해당
        high_freq_start = int(freq_dim * 0.7)  # 고주파 범위. 약 4kHz 이상에 해당
        
        # 교란을 저주파, 중주파, 고주파 부분으로 분리
        if perturbation.dim() == 3:
            low_freq = perturbation[:, :low_freq_end, :]
            mid_freq = perturbation[:, low_freq_end:high_freq_start, :]
            high_freq = perturbation[:, high_freq_start:, :]
        else:
            low_freq = perturbation[:, :, :low_freq_end, :]
            mid_freq = perturbation[:, :, low_freq_end:high_freq_start, :]
            high_freq = perturbation[:, :, high_freq_start:, :]
        
        # 서로 다른 제약 적용
        low_freq_constrained = torch.clamp(low_freq, -epsilon1, epsilon1)
        mid_freq_constrained = torch.clamp(mid_freq, -epsilon2, epsilon2)
        high_freq_constrained = torch.clamp(high_freq, -epsilon3, epsilon3)
        
        # 교란 재구성
        weighted_perturbation = torch.cat(
            [low_freq_constrained, mid_freq_constrained, high_freq_constrained],
            dim=cat_dim
        )
        
        return weighted_perturbation


def waveform_l2_loss(original: torch.Tensor, protected: torch.Tensor) -> torch.Tensor:
    """
    수정: perceptual-aware optimization 1단계 품질 제약.
    원본 waveform과 보호 waveform 사이의 직접적인 L2 차이를 계산한다.

    Args:
        original: 원본 waveform [..., T]
        protected: 보호 waveform [..., T]

    Returns:
        L2 waveform loss
    """
    min_len = min(original.shape[-1], protected.shape[-1])
    original = original[..., :min_len]
    protected = protected[..., :min_len]

    return torch.mean((original - protected) ** 2)


def stft_loss(original: torch.Tensor,
              protected: torch.Tensor,
              n_fft: int = 1024,
              hop_length: int = 256,
              win_length: Optional[int] = None) -> torch.Tensor:
    """
    수정: perceptual-aware optimization 2단계 품질 제약.
    STFT magnitude 영역에서 원본과 보호 음성의 차이를 계산한다.

    Args:
        original: 원본 waveform [1, T] 또는 [B, T]
        protected: 보호 waveform [1, T] 또는 [B, T]
        n_fft: FFT 크기
        hop_length: 프레임 이동 간격
        win_length: window 길이. None이면 n_fft 사용

    Returns:
        STFT magnitude loss
    """
    if win_length is None:
        win_length = n_fft

    min_len = min(original.shape[-1], protected.shape[-1])
    original = original[..., :min_len]
    protected = protected[..., :min_len]

    window = torch.hann_window(win_length, device=original.device)

    original_stft = torch.stft(
        original,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True
    )

    protected_stft = torch.stft(
        protected,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True
    )

    original_mag = torch.abs(original_stft)
    protected_mag = torch.abs(protected_stft)

    return torch.mean(torch.abs(original_mag - protected_mag))


def multi_resolution_stft_loss(original: torch.Tensor,
                               protected: torch.Tensor,
                               resolutions: Optional[List[Tuple[int, int, int]]] = None) -> torch.Tensor:
    """
    수정: perceptual-aware optimization 2단계 확장.
    여러 STFT 해상도에서 스펙트럼 왜곡을 측정한다.

    Args:
        original: 원본 waveform [1, T] 또는 [B, T]
        protected: 보호 waveform [1, T] 또는 [B, T]
        resolutions: (n_fft, hop_length, win_length) 목록

    Returns:
        multi-resolution STFT loss
    """
    if resolutions is None:
        resolutions = [
            (512, 128, 512),
            (1024, 256, 1024),
            (2048, 512, 2048),
        ]

    losses = []
    for n_fft, hop_length, win_length in resolutions:
        losses.append(
            stft_loss(
                original,
                protected,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length
            )
        )

    return torch.stack(losses).mean()


def mel_distance_loss(original: torch.Tensor,
                      protected: torch.Tensor,
                      converter: MelSpectrogramConverter) -> torch.Tensor:
    """
    수정: perceptual-aware optimization 3단계 품질 제약.
    mel-spectrogram 영역에서 원본 음성과 보호 음성의 차이를 계산한다.

    Args:
        original: 원본 waveform [1, T] 또는 [B, T]
        protected: 보호 waveform [1, T] 또는 [B, T]
        converter: MelSpectrogramConverter 인스턴스

    Returns:
        mel-spectrogram distance loss
    """
    min_len = min(original.shape[-1], protected.shape[-1])
    original = original[..., :min_len]
    protected = protected[..., :min_len]

    original_mel = converter.waveform_to_mel(original)
    protected_mel = converter.waveform_to_mel(protected)

    min_frames = min(original_mel.shape[-1], protected_mel.shape[-1])
    original_mel = original_mel[..., :min_frames]
    protected_mel = protected_mel[..., :min_frames]

    return torch.mean(torch.abs(original_mel - protected_mel))


def apply_random_shift(waveform: torch.Tensor, max_shift: int = 100) -> torch.Tensor:
    """
    파형에 무작위 이동 적용
    
    Args:
        waveform: 입력 파형 [1, T]
        max_shift: 최대 이동량
        
    Returns:
        이동된 파형 [1, T]
    """
    shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
    
    if shift > 0:
        # 오른쪽으로 이동
        shifted_waveform = torch.cat([
            torch.zeros(1, shift, device=waveform.device),
            waveform[:, :-shift]
        ], dim=1)
    elif shift < 0:
        # 왼쪽으로 이동
        shifted_waveform = torch.cat([
            waveform[:, -shift:],
            torch.zeros(1, -shift, device=waveform.device)
        ], dim=1)
    else:
        shifted_waveform = waveform
    
    return shifted_waveform