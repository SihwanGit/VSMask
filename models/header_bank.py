import os
from typing import Optional, Tuple

import torch


class HeaderBank:
    def __init__(
        self,
        num_headers: int = 5,
        mel_bins: int = 80,
        time_length: int = 100,
        device: str = "cuda",
    ):
        """
        Header Bank 모델

        여러 개의 universal perturbation header를 보관하고,
        입력 발화 시작 조건에 따라 하나의 header를 선택해 적용한다.

        Args:
            num_headers: Header Bank에 포함할 header 개수
            mel_bins: mel-spectrogram 주파수 차원
            time_length: header 시간 길이. mel frame 단위
            device: 사용할 장치
        """
        # 수정: CUDA가 없는 환경에서 device='cuda'가 들어오면 자동으로 CPU 사용
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA를 사용할 수 없습니다. HeaderBank의 device를 cpu로 변경합니다.")
            device = "cpu"

        if num_headers <= 0:
            raise ValueError("num_headers는 1 이상이어야 합니다.")

        self.num_headers = num_headers
        self.mel_bins = mel_bins
        self.time_length = time_length
        self.device = device

        # 수정: fixed header 1개가 아니라 K개의 header를 학습/저장하기 위한 구조
        # shape: [K, 1, F, T]
        self.headers = torch.zeros(
            (num_headers, 1, mel_bins, time_length),
            device=device,
            requires_grad=True,
        )

    def parameters(self):
        """
        optimizer에 넘길 학습 대상 파라미터를 반환한다.
        """
        return [self.headers]

    def _validate_header_index(self, header_index: int) -> None:
        """
        header index가 유효한지 검사한다.
        """
        if not isinstance(header_index, int):
            raise TypeError(f"header_index는 int여야 합니다. 현재 타입: {type(header_index)}")

        if header_index < 0 or header_index >= self.num_headers:
            raise IndexError(
                f"header_index 범위가 잘못되었습니다: {header_index}. "
                f"가능 범위: 0 ~ {self.num_headers - 1}"
            )

    def get_header(self, header_index: int) -> torch.Tensor:
        """
        선택된 header를 반환한다.

        Args:
            header_index: 사용할 header index

        Returns:
            선택된 header [1, 1, F, T]
        """
        self._validate_header_index(header_index)

        # [1, F, T] -> [1, 1, F, T]
        return self.headers[header_index:header_index + 1]

    def _match_header_to_input(
        self,
        source_mel: torch.Tensor,
        header: torch.Tensor,
    ) -> Tuple[torch.Tensor, int, int]:
        """
        입력 mel과 header의 주파수/시간 차원이 다를 수 있으므로,
        공통으로 사용할 수 있는 영역만 반환한다.

        Args:
            source_mel: 입력 mel [B, 1, F, T]
            header: 선택된 header [1, 1, Fh, Th]

        Returns:
            cropped_header: 입력 mel에 적용 가능한 header
            freq_len: 적용할 주파수 길이
            time_len: 적용할 시간 길이
        """
        if source_mel.dim() != 4:
            raise ValueError(
                f"source_mel은 [B, 1, F, T] 형태여야 합니다. 현재 shape: {tuple(source_mel.shape)}"
            )

        if header.dim() != 4:
            raise ValueError(
                f"header는 [1, 1, F, T] 형태여야 합니다. 현재 shape: {tuple(header.shape)}"
            )

        freq_len = min(source_mel.shape[2], header.shape[2])
        time_len = min(source_mel.shape[3], header.shape[3])

        cropped_header = header[:, :, :freq_len, :time_len]

        return cropped_header, freq_len, time_len

    def apply_header(self, source_mel: torch.Tensor, header_index: int) -> torch.Tensor:
        """
        선택된 header를 입력 mel 앞부분에 적용한다.

        Args:
            source_mel: 원본 mel-spectrogram [B, 1, F, T]
            header_index: 적용할 header index

        Returns:
            header가 적용된 mel-spectrogram [B, 1, F, T]
        """
        source_mel = source_mel.to(self.device)

        header = self.get_header(header_index)
        header, freq_len, time_len = self._match_header_to_input(source_mel, header)

        perturbed_mel = source_mel.clone()
        perturbed_mel[:, :, :freq_len, :time_len] += header

        perturbed_mel = torch.clamp(perturbed_mel, -1.0, 1.0)

        return perturbed_mel

    def initialize_from_fixed_header(self, fixed_header_path: str, noise_scale: float = 0.001) -> None:
        """
        기존 fixed universal header를 Header Bank의 모든 header 초기값으로 복제한다.

        Args:
            fixed_header_path: 기존 universal_header.pt 경로
            noise_scale: 각 header를 조금 다르게 만들기 위한 작은 noise 크기
        """
        fixed_header = torch.load(fixed_header_path, map_location=self.device)

        if not isinstance(fixed_header, torch.Tensor):
            raise TypeError(
                f"fixed header가 torch.Tensor가 아닙니다. 현재 타입: {type(fixed_header)}"
            )

        if fixed_header.dim() != 4:
            raise ValueError(
                f"fixed header는 [1, 1, F, T] 형태여야 합니다. 현재 shape: {tuple(fixed_header.shape)}"
            )

        freq_len = min(self.mel_bins, fixed_header.shape[2])
        time_len = min(self.time_length, fixed_header.shape[3])

        with torch.no_grad():
            self.headers.zero_()

            for i in range(self.num_headers):
                self.headers[i:i + 1, :, :freq_len, :time_len] = fixed_header[:, :, :freq_len, :time_len]

                # 수정: 모든 header가 완전히 같으면 bank 의미가 약하므로 작은 noise를 추가
                if noise_scale > 0:
                    self.headers[i:i + 1] += noise_scale * torch.randn_like(self.headers[i:i + 1])

            self.headers.data = torch.clamp(self.headers.data, -0.1, 0.1)

        self.headers.requires_grad = True

    def save(self, path: str) -> None:
        """
        Header Bank를 파일에 저장한다.
        """
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        payload = {
            "num_headers": self.num_headers,
            "mel_bins": self.mel_bins,
            "time_length": self.time_length,
            "headers": self.headers.detach().cpu(),
        }

        torch.save(payload, path)

    def load(self, path: str) -> None:
        """
        파일에서 Header Bank를 로드한다.
        """
        payload = torch.load(path, map_location=self.device)

        if isinstance(payload, torch.Tensor):
            # 수정: 예외적으로 tensor만 저장된 경우도 로드 가능하게 처리
            loaded_headers = payload
            if loaded_headers.dim() != 4:
                raise ValueError(
                    f"로드한 headers는 [K, 1, F, T] 형태여야 합니다. 현재 shape: {tuple(loaded_headers.shape)}"
                )

            self.headers = loaded_headers.to(self.device)
            self.num_headers = self.headers.shape[0]
            self.mel_bins = self.headers.shape[2]
            self.time_length = self.headers.shape[3]
            self.headers.requires_grad = True
            return

        if not isinstance(payload, dict):
            raise TypeError(f"지원하지 않는 Header Bank 저장 형식입니다. 현재 타입: {type(payload)}")

        required_keys = {"num_headers", "mel_bins", "time_length", "headers"}
        missing_keys = required_keys - set(payload.keys())
        if missing_keys:
            raise KeyError(f"Header Bank 파일에 필요한 key가 없습니다: {missing_keys}")

        loaded_headers = payload["headers"]

        if not isinstance(loaded_headers, torch.Tensor):
            raise TypeError(
                f"payload['headers']가 torch.Tensor가 아닙니다. 현재 타입: {type(loaded_headers)}"
            )

        if loaded_headers.dim() != 4:
            raise ValueError(
                f"headers는 [K, 1, F, T] 형태여야 합니다. 현재 shape: {tuple(loaded_headers.shape)}"
            )

        self.num_headers = int(payload["num_headers"])
        self.mel_bins = int(payload["mel_bins"])
        self.time_length = int(payload["time_length"])
        self.headers = loaded_headers.to(self.device)
        self.headers.requires_grad = True