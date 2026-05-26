import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


"""
핵심 수정 사항
- CUDA가 없는 환경에서 device='cuda'가 들어와도 CPU로 자동 전환.
- source_mel, target_mel, header의 주파수/시간 shape가 다를 때 안전하게 crop.
- source_embedding, target_embedding은 매 iteration마다 변하지 않으므로 미리 계산하고 detach().
- save()에서 저장 폴더가 없으면 자동 생성.
- load()에서 불러온 header를 현재 device로 올리고 requires_grad=True 유지.
- apply_header()에서 변수명 F를 피해서 torch.nn.functional as F와 혼동되지 않게 수정.
"""

class UniversalPerturbationHeader:
    def __init__(self, mel_bins: int = 80, time_length: int = 100, device: str = 'cuda'):
        """
        범용 교란 헤더 모델
        
        Args:
            mel_bins: 멜 스펙트로그램의 주파수 차원
            time_length: 시간 차원 길이
            device: 사용할 장치
        """
        # 수정: CUDA가 없는 환경에서 device='cuda'가 들어오면 자동으로 CPU 사용
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA를 사용할 수 없습니다. UniversalPerturbationHeader의 device를 cpu로 변경합니다.")
            device = 'cpu'

        self.mel_bins = mel_bins
        self.time_length = time_length
        self.device = device
        
        # 범용 교란 헤더 초기화
        self.header = torch.zeros((1, 1, mel_bins, time_length), device=device)
        self.header.requires_grad = True

    def _match_header_to_input(self, source_mel: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        수정: source_mel과 header의 주파수/시간 차원이 다를 수 있으므로,
        둘이 공통으로 사용할 수 있는 영역만 잘라서 반환한다.

        Args:
            source_mel: 원본 화자의 멜 스펙트로그램 [B, 1, F, T]

        Returns:
            cropped_header: source_mel에 적용 가능한 크기로 잘린 header
            freq_len: 적용할 주파수 차원 길이
            time_len: 적용할 시간 차원 길이
        """
        if source_mel.dim() != 4:
            raise ValueError(
                f"source_mel은 [B, 1, F, T] 형태여야 합니다. 현재 shape: {tuple(source_mel.shape)}"
            )

        freq_len = min(source_mel.shape[2], self.header.shape[2])
        time_len = min(source_mel.shape[3], self.header.shape[3])

        cropped_header = self.header[:, :, :freq_len, :time_len]

        return cropped_header, freq_len, time_len

    def optimize(self, source_mel: torch.Tensor, target_mel: torch.Tensor,
                 speaker_encoder, optimizer, num_iterations: int = 1000,
                 epsilon: float = 0.1, lambda_param: float = 0.5) -> None:
        """
        범용 교란 헤더 최적화
        
        Args:
            source_mel: 원본 화자의 멜 스펙트로그램 [B, 1, F, T]
            target_mel: 목표 화자의 멜 스펙트로그램 [B, 1, F, T]
            speaker_encoder: 화자 인코더 모델
            optimizer: 옵티마이저
            num_iterations: 최적화 반복 횟수
            epsilon: 교란의 최대 크기
            lambda_param: 손실 함수에서 사용하는 균형 파라미터
        """
        # 수정: 입력 텐서를 header와 같은 device로 이동
        source_mel = source_mel.to(self.device)
        target_mel = target_mel.to(self.device)
        speaker_encoder = speaker_encoder.to(self.device)

        # 수정: speaker_encoder 자체는 학습하지 않고 header만 학습하므로 eval 모드로 설정
        speaker_encoder.eval()

        # 수정: source_mel, target_mel, header의 shape가 다를 수 있으므로 공통 영역만 사용
        header, freq_len, time_len = self._match_header_to_input(source_mel)
        source_mel_used = source_mel[:, :, :freq_len, :time_len]
        target_mel_used = target_mel[:, :, :freq_len, :time_len]

        # 수정: source_embedding과 target_embedding은 iteration마다 변하지 않으므로 미리 계산
        #       detach()를 통해 speaker_encoder 쪽으로 불필요한 그래프가 쌓이지 않도록 함
        with torch.no_grad():
            source_embedding = speaker_encoder(source_mel_used).detach()
            target_embedding = speaker_encoder(target_mel_used).detach()

        for i in range(num_iterations):
            # 수정: 매 반복마다 최신 self.header에서 공통 영역을 다시 가져옴
            #       optimizer.step() 이후 header 값이 바뀌기 때문
            header, freq_len, time_len = self._match_header_to_input(source_mel)
            source_mel_used = source_mel[:, :, :freq_len, :time_len]

            # 교란 추가
            perturbed_mel = source_mel_used + header
            
            # 교란 크기 제한
            perturbed_mel = torch.clamp(perturbed_mel, -1.0, 1.0)
            
            # 화자 임베딩 계산
            # 수정: source_embedding, target_embedding은 위에서 미리 계산했고,
            #       여기서는 header에 의해 바뀌는 perturbed_embedding만 계산
            perturbed_embedding = speaker_encoder(perturbed_mel)
            
            # 손실 계산: 목표 화자와의 거리는 최소화하고, 원본 화자와의 거리는 최대화
            loss_target = F.mse_loss(perturbed_embedding, target_embedding)
            loss_source = F.mse_loss(perturbed_embedding, source_embedding)
            
            loss = loss_target - lambda_param * loss_source
            
            # 그래디언트 업데이트
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 교란 크기 제한
            with torch.no_grad():
                self.header.data = torch.clamp(self.header.data, -epsilon, epsilon)
            
            if (i + 1) % 100 == 0 or (i + 1) == num_iterations:
                print(f"Iteration {i+1}/{num_iterations}, Loss: {loss.item():.6f}")
    
    def apply_header(self, source_mel: torch.Tensor) -> torch.Tensor:
        """
        입력 멜 스펙트로그램에 범용 교란 헤더 적용
        
        Args:
            source_mel: 원본 화자의 멜 스펙트로그램 [B, 1, F, T]
            
        Returns:
            교란 헤더가 추가된 멜 스펙트로그램 [B, 1, F, T]
        """
        # 수정: 입력 텐서를 header와 같은 device로 이동
        source_mel = source_mel.to(self.device)

        # 교란 헤더와 입력 형태가 호환되는지 확인
        # 수정: 변수명 F는 torch.nn.functional as F와 혼동될 수 있으므로 freq_dim으로 변경
        _, _, freq_dim, time_dim = source_mel.shape

        # 수정: 입력과 header의 주파수/시간 차원이 다를 수 있으므로 공통 영역만 사용
        header, freq_len, time_len = self._match_header_to_input(source_mel)
        
        # 입력이 교란 헤더보다 짧으면 교란 헤더를 잘라냄
        # 수정: _match_header_to_input()에서 이미 시간 길이를 잘라내므로 별도 슬라이싱 불필요
        
        # 입력이 교란 헤더보다 길면 앞부분에만 교란 추가
        perturbed_mel = source_mel.clone()
        perturbed_mel[:, :, :freq_len, :time_len] += header
        
        # 결과 범위 제한
        perturbed_mel = torch.clamp(perturbed_mel, -1.0, 1.0)
        
        return perturbed_mel
    
    def save(self, path: str) -> None:
        """범용 교란 헤더를 파일에 저장"""
        # 수정: 저장 경로의 디렉터리가 없으면 자동 생성
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        # 수정: 저장 시 CPU로 옮겨 저장하면 다른 환경에서도 로드하기 쉬움
        torch.save(self.header.detach().cpu(), path)
    
    def load(self, path: str) -> None:
        """파일에서 범용 교란 헤더 로드"""
        # 수정: 현재 device에 맞춰 로드
        loaded_header = torch.load(path, map_location=self.device)

        # 수정: 혹시 저장된 객체가 Tensor가 아닌 경우를 방어
        if not isinstance(loaded_header, torch.Tensor):
            raise TypeError(
                f"로드한 header가 torch.Tensor가 아닙니다. 현재 타입: {type(loaded_header)}"
            )

        # 수정: 로드한 header를 현재 device로 이동하고 학습 가능 상태로 설정
        self.header = loaded_header.to(self.device)
        self.header.requires_grad = True

        # 수정: 로드된 header 기준으로 mel_bins, time_length 갱신
        if self.header.dim() != 4:
            raise ValueError(
                f"header는 [1, 1, F, T] 형태여야 합니다. 현재 shape: {tuple(self.header.shape)}"
            )

        self.mel_bins = self.header.shape[2]
        self.time_length = self.header.shape[3]