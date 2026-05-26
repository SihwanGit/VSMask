import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional


class DownSamplingBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int],
                 stride: Tuple[int, int]):
        """
        다운샘플링 모듈
        
        Args:
            in_channels: 입력 채널 수
            out_channels: 출력 채널 수
            kernel_size: 커널 크기
            stride: 스트라이드
        """
        super(DownSamplingBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.ReflectionPad2d((kernel_size[0]//2, kernel_size[0]//2,
                               kernel_size[1]//2, kernel_size[1]//2)),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpSamplingBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int],
                 stride: Tuple[int, int]):
        """
        업샘플링 모듈
        
        Args:
            in_channels: 입력 채널 수
            out_channels: 출력 채널 수
            kernel_size: 커널 크기
            stride: 스트라이드
        """
        super(UpSamplingBlock, self).__init__()
        
        self.conv_transpose = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_transpose(x)


class PredictiveModel(nn.Module):
    def __init__(self, mel_bins: int = 80, time_dim: int = 100,
                 output_time_dim: int = 25, min_input_time_dim: int = 64):
        """
        VSMask 예측 모델
        
        Args:
            mel_bins: 멜 스펙트로그램의 주파수 차원
            time_dim: 입력 시간 차원
            output_time_dim: 예측 교란의 출력 시간 차원
            min_input_time_dim: 다운샘플링 안정성을 위한 최소 입력 시간 차원
        """
        super(PredictiveModel, self).__init__()

        self.mel_bins = mel_bins
        self.time_dim = time_dim

        # 수정: VSMask 논문에서 출력 perturbation 길이 γ=0.4초에 해당하는 값을 명시적으로 둔다.
        # sample_rate=16000, hop_length=256 기준 0.4초는 약 25 mel frames이다.
        self.output_time_dim = output_time_dim

        # 수정: 입력 mel 길이가 너무 짧으면 반복 다운샘플링 과정에서 시간축이 0에 가까워질 수 있으므로 최소 길이를 둔다.
        self.min_input_time_dim = min_input_time_dim
        
        # 다운샘플링 네트워크 구조 정의
        self.down_blocks = nn.ModuleList([
            DownSamplingBlock(1, 32, (3, 3), (1, 2)),
            DownSamplingBlock(32, 64, (3, 3), (2, 2)),
            DownSamplingBlock(64, 128, (3, 3), (2, 2)),
            DownSamplingBlock(128, 256, (3, 3), (2, 2)),
            DownSamplingBlock(256, 256, (3, 3), (2, 2)),
            DownSamplingBlock(256, 512, (3, 3), (2, 2)),
            DownSamplingBlock(512, 512, (3, 3), (2, 2))
        ])
        
        # 업샘플링 네트워크 구조 정의
        self.up_blocks = nn.ModuleList([
            UpSamplingBlock(512, 256, (3, 3), (2, 2)),
            UpSamplingBlock(256, 128, (3, 3), (2, 2)),
            UpSamplingBlock(128, 64, (3, 3), (2, 2)),
            UpSamplingBlock(64, 32, (3, 3), (2, 2)),
            UpSamplingBlock(32, 1, (3, 3), (2, 2))
        ])
        
        # 최종 출력을 위한 tanh 활성화 함수
        self.tanh = nn.Tanh()

    def _validate_input(self, x: torch.Tensor) -> None:
        """
        수정: 입력 shape를 명확히 검사해 shape mismatch 원인을 빠르게 찾을 수 있게 한다.
        """
        if x.dim() != 4:
            raise ValueError(
                f"PredictiveModel 입력은 [B, 1, F, T] 형태여야 합니다. 현재 shape: {tuple(x.shape)}"
            )

        if x.shape[1] != 1:
            raise ValueError(
                f"PredictiveModel 입력 channel 차원은 1이어야 합니다. 현재 channel: {x.shape[1]}"
            )

    def _pad_input_if_too_short(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        수정: 입력 시간축이 너무 짧으면 오른쪽 zero-padding을 적용한다.
        다운샘플링이 여러 번 반복되므로 짧은 smoke test 입력에서 오류가 나는 것을 줄인다.

        Returns:
            padded_x: padding이 적용된 입력
            original_time_dim: 원래 시간축 길이
        """
        original_time_dim = x.shape[-1]

        if original_time_dim >= self.min_input_time_dim:
            return x, original_time_dim

        pad_len = self.min_input_time_dim - original_time_dim
        x = F.pad(x, (0, pad_len, 0, 0))

        return x, original_time_dim

    def _resize_output(self, x: torch.Tensor, target_freq_dim: int) -> torch.Tensor:
        """
        수정: ConvTranspose2d를 거친 출력 shape가 입력 mel shape와 정확히 맞지 않을 수 있다.
        최종 perturbation을 [B, 1, target_freq_dim, output_time_dim]으로 보정한다.

        이 처리는 baseline 실행 안정성을 위한 것이며,
        논문 수식의 핵심 목적식 자체를 바꾸는 것은 아니다.
        """
        x = F.interpolate(
            x,
            size=(target_freq_dim, self.output_time_dim),
            mode='bilinear',
            align_corners=False
        )

        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파
        
        Args:
            x: 입력 멜 스펙트로그램 [B, 1, F, T]
            
        Returns:
            예측된 교란 [B, 1, F, T']
        """
        # 수정: 입력 shape를 먼저 검사
        self._validate_input(x)

        # 수정: 출력 주파수 차원을 입력 mel의 주파수 차원과 맞추기 위해 저장
        target_freq_dim = x.shape[2]

        # 수정: 짧은 입력에서 downsampling이 깨지는 것을 방지하기 위해 필요한 경우 padding
        x, original_time_dim = self._pad_input_if_too_short(x)

        # 다운샘플링 부분
        down_features = []
        for block in self.down_blocks:
            x = block(x)
            down_features.append(x)
        
        # 업샘플링 부분
        for i, block in enumerate(self.up_blocks):
            x = block(x)

        # 수정: 업샘플링 결과의 F/T shape가 입력과 다를 수 있으므로 최종 출력 shape를 보정
        x = self._resize_output(x, target_freq_dim=target_freq_dim)
        
        # tanh 활성화 함수를 적용하여 출력 정규화
        x = self.tanh(x)
        
        return x