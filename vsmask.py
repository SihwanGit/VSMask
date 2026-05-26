import os
import argparse
import torch
import numpy as np
import soundfile as sf
import torchaudio
import tqdm
from typing import Optional, Tuple

from models.predictive_model import PredictiveModel
from models.header_model import UniversalPerturbationHeader
from utils.audio import MelSpectrogramConverter


def move_converter_to_device(converter: MelSpectrogramConverter, device: str) -> MelSpectrogramConverter:
    """
    수정: MelSpectrogramConverter 내부 torchaudio transform들을 현재 device로 이동한다.
    waveform은 cuda에 있는데 mel_transform이 cpu에 있으면 device mismatch가 발생할 수 있다.
    """
    converter.mel_transform = converter.mel_transform.to(device)
    converter.inverse_mel_transform = converter.inverse_mel_transform.to(device)
    converter.griffin_lim = converter.griffin_lim.to(device)
    return converter


def apply_weighted_constraint_4d(perturbation: torch.Tensor,
                                 epsilon1: float = 0.1,
                                 epsilon2: float = 0.05,
                                 epsilon3: float = 0.08) -> torch.Tensor:
    """
    수정: vsmask.py에서 사용하는 perturbation은 [B, 1, F, T] 형태의 4차원 텐서다.
    기존 utils/audio.py의 apply_weighted_constraint()는 [1, F, T] 형태를 가정하므로,
    여기서는 4차원 입력에 직접 가중치 기반 제약을 적용한다.

    Args:
        perturbation: 멜 스펙트로그램 교란 [B, 1, F, T]
        epsilon1: 저주파 교란의 최대 크기
        epsilon2: 중주파 교란의 최대 크기
        epsilon3: 고주파 교란의 최대 크기

    Returns:
        가중치 기반 제약이 적용된 교란 [B, 1, F, T]
    """
    if perturbation.dim() != 4:
        raise ValueError(
            f"apply_weighted_constraint_4d는 [B, 1, F, T] 형태를 기대합니다. "
            f"현재 shape: {tuple(perturbation.shape)}"
        )

    _, _, freq_dim, _ = perturbation.shape

    low_freq_end = int(freq_dim * 0.3)
    high_freq_start = int(freq_dim * 0.7)

    low_freq = perturbation[:, :, :low_freq_end, :]
    mid_freq = perturbation[:, :, low_freq_end:high_freq_start, :]
    high_freq = perturbation[:, :, high_freq_start:, :]

    low_freq_constrained = torch.clamp(low_freq, -epsilon1, epsilon1)
    mid_freq_constrained = torch.clamp(mid_freq, -epsilon2, epsilon2)
    high_freq_constrained = torch.clamp(high_freq, -epsilon3, epsilon3)

    return torch.cat(
        [low_freq_constrained, mid_freq_constrained, high_freq_constrained],
        dim=2
    )


def crop_perturbation_to_mel(predicted_perturbation: torch.Tensor,
                             reference_mel: torch.Tensor) -> torch.Tensor:
    """
    수정: PredictiveModel의 출력 주파수/시간 크기가 reference_mel과 정확히 일치하지 않을 수 있다.
    inference 단계에서 shape mismatch를 피하기 위해 적용 가능한 범위로 crop한다.

    Args:
        predicted_perturbation: 예측된 교란 [B, 1, Fp, Tp]
        reference_mel: 기준 멜 스펙트로그램 [B, 1, Fr, Tr]

    Returns:
        reference_mel에 적용 가능한 크기로 잘린 교란 [B, 1, min(Fp, Fr), min(Tp, Tr)]
    """
    freq_len = min(reference_mel.shape[2], predicted_perturbation.shape[2])
    time_len = min(reference_mel.shape[3], predicted_perturbation.shape[3])

    return predicted_perturbation[:, :, :freq_len, :time_len]


class VSMask:
    def __init__(self, predictive_model_path: str, header_path: str,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 sample_rate: int = 16000, n_fft: int = 1024,
                 hop_length: int = 256, n_mels: int = 80):
        """
        VSMask 시스템 클래스
        
        Args:
            predictive_model_path: 예측 모델 경로
            header_path: 범용 교란 헤더 경로
            device: 사용할 장치
        """
        # 수정: CUDA가 없는 환경에서 device='cuda'가 들어오면 자동으로 CPU 사용
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA를 사용할 수 없습니다. device를 cpu로 변경합니다.")
            device = 'cpu'

        self.device = device

        # 수정: train_header.py / train_predictive.py와 동일한 mel 설정을 사용할 수 있도록 저장
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        # 예측 모델 로드
        self.predictive_model = PredictiveModel(
            mel_bins=n_mels,
            time_dim=78  # 수정: 현재 PredictiveModel은 time_dim을 실제로 사용하지 않지만, 논문 t=1.25초 ≈ 78 frames 기준값 명시
        ).to(device)

        # 수정: torch.load 결과가 CPU/GPU 어디서 저장됐든 현재 device에 맞춰 로드
        self.predictive_model.load_state_dict(
            torch.load(predictive_model_path, map_location=device)
        )
        self.predictive_model.eval()
        
        # 범용 교란 헤더 로드
        self.header = UniversalPerturbationHeader(
            mel_bins=n_mels,
            time_length=100,
            device=device
        )
        self.header.load(header_path)
        
        # 오디오 변환기 초기화
        # 수정: 기본값 고정 대신 CLI에서 받은 sample_rate/n_fft/hop_length/n_mels 사용
        self.converter = MelSpectrogramConverter(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

        # 수정: converter 내부 transform을 현재 device로 이동
        self.converter = move_converter_to_device(self.converter, device)
        
        print(f"VSMask 시스템이 초기화되었습니다. 사용 장치: {device}")
        print("[설정 확인]")
        print(f"  sample_rate : {self.sample_rate}")
        print(f"  n_fft       : {self.n_fft}")
        print(f"  hop_length  : {self.hop_length}")
        print(f"  n_mels      : {self.n_mels}")
    
    def protect_file(self, input_path: str, output_path: str,
                     window_size: int = 78, future_step: int = 25,
                     epsilon1: float = 0.115, epsilon2: float = 0.085, epsilon3: float = 0.1) -> None:
        """
        오디오 파일 보호
        
        Args:
            input_path: 입력 오디오 파일 경로
            output_path: 출력 오디오 파일 경로
            window_size: 슬라이딩 윈도우 크기
            future_step: 예측할 미래 시간 스텝 수
            epsilon1: 저주파 교란의 최대 크기
            epsilon2: 중주파 교란의 최대 크기
            epsilon3: 고주파 교란의 최대 크기
        """
        # 수정: 보호 파라미터가 mel frame 기준임을 출력하여 단위 혼동 방지
        print("[보호 파라미터]")
        print(f"  window_size : {window_size} mel frames ({window_size * self.hop_length / self.sample_rate:.3f} sec)")
        print(f"  future_step : {future_step} mel frames ({future_step * self.hop_length / self.sample_rate:.3f} sec)")
        print(f"  epsilon1    : {epsilon1}")
        print(f"  epsilon2    : {epsilon2}")
        print(f"  epsilon3    : {epsilon3}")

        # 오디오 파일 로드
        waveform, sample_rate = torchaudio.load(input_path)
        
        # 모델에서 요구하는 샘플링 레이트로 리샘플링. 필요한 경우에만 수행
        if sample_rate != self.converter.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.converter.sample_rate)
            waveform = resampler(waveform)
            sample_rate = self.converter.sample_rate
        
        # 스테레오인 경우 모노로 변환
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 수정: waveform 값 범위 방어
        waveform = torch.clamp(waveform, -1.0, 1.0)
        
        # 파형을 장치로 이동
        waveform = waveform.to(self.device)
        
        # 보호 적용
        protected_waveform = self._protect_waveform(
            waveform, window_size, future_step,
            epsilon1, epsilon2, epsilon3
        )

        # 수정: Griffin-Lim 복원 결과가 원본보다 길거나 짧을 수 있으므로 원본 길이에 맞춤
        protected_waveform = self._match_waveform_length(protected_waveform, waveform.shape[-1])

        # 수정: 저장 전 값 범위 제한
        protected_waveform = torch.clamp(protected_waveform, -1.0, 1.0)

        # 수정: 출력 폴더가 없으면 자동 생성
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 보호된 오디오 저장
        torchaudio.save(output_path, protected_waveform.cpu(), sample_rate)
        print(f"보호된 오디오가 {output_path}에 저장되었습니다.")
    
    def protect_stream(self, input_stream, output_stream,
                     window_size: int = 78, future_step: int = 25,
                     epsilon1: float = 0.115, epsilon2: float = 0.085, epsilon3: float = 0.1) -> None:
        """
        오디오 스트림 보호. 실시간 적용
        
        Args:
            input_stream: 입력 오디오 스트림
            output_stream: 출력 오디오 스트림
            window_size: 슬라이딩 윈도우 크기
            future_step: 예측할 미래 시간 스텝 수
            epsilon1: 저주파 교란의 최대 크기
            epsilon2: 중주파 교란의 최대 크기
            epsilon3: 고주파 교란의 최대 크기
        """
        # 예시 코드이며, 실제 사용 시에는 구체적인 스트림 처리 로직에 맞게 수정해야 함
        buffer = []
        header_applied = False
        
        while True:
            # 오디오 청크 읽기
            audio_chunk = input_stream.read(window_size)
            if not audio_chunk:
                break
            
            # torch 텐서로 변환
            chunk_tensor = torch.tensor(audio_chunk, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # 아직 header를 적용하지 않았다면 먼저 적용
            if not header_applied:
                # 멜 스펙트로그램으로 변환
                chunk_mel = self.converter.waveform_to_mel(chunk_tensor)

                # 수정: waveform_to_mel이 [1, F, T]를 반환하는 경우 [B, 1, F, T]로 맞춤
                if chunk_mel.dim() == 3:
                    chunk_mel = chunk_mel.unsqueeze(0)
                
                # header 적용
                # 수정: header와 chunk_mel의 주파수/시간 shape가 다를 수 있으므로 안전하게 crop
                header_freq = min(chunk_mel.shape[2], self.header.header.shape[2])
                header_length = min(chunk_mel.shape[-1], self.header.header.shape[-1])
                chunk_mel[:, :, :header_freq, :header_length] += \
                    self.header.header[:, :, :header_freq, :header_length]
                
                # 다시 파형으로 변환
                protected_chunk = self.converter.mel_to_waveform(chunk_mel.squeeze(0))
                header_applied = True
            else:
                # 예측 모델을 이용한 보호 적용
                # 슬라이딩 윈도우 구성
                buffer.append(chunk_tensor)
                if len(buffer) > max(1, window_size // chunk_tensor.shape[1]):
                    buffer.pop(0)
                
                window = torch.cat(buffer, dim=1)
                
                # 멜 스펙트로그램으로 변환
                window_mel = self.converter.waveform_to_mel(window)

                # 수정: waveform_to_mel이 [1, F, T]를 반환하는 경우 [B, 1, F, T]로 맞춤
                if window_mel.dim() == 3:
                    window_mel = window_mel.unsqueeze(0)
                
                # 교란 예측
                with torch.no_grad():
                    perturbation = self.predictive_model(window_mel)

                # 수정: 예측 교란 shape를 window_mel에 적용 가능한 범위로 crop
                perturbation = crop_perturbation_to_mel(perturbation, window_mel)
                
                # 미래 시간 스텝에 교란 적용
                future_mel = window_mel.clone()
                future_idx = future_step
                future_end = min(future_idx + perturbation.shape[-1], future_mel.shape[-1])

                if future_idx < future_mel.shape[-1]:
                    # 수정: 주파수 차원도 공통 범위만 사용
                    freq_len = min(future_mel.shape[2], perturbation.shape[2])
                    apply_time_len = future_end - future_idx

                    future_mel[:, :, :freq_len, future_idx:future_end] += \
                        perturbation[:, :, :freq_len, :apply_time_len]
                
                # 가중치 기반 제약 적용
                # 수정: 4D 입력을 처리하는 apply_weighted_constraint_4d 사용
                weighted_perturbed = apply_weighted_constraint_4d(
                    future_mel - window_mel,
                    epsilon1=epsilon1,
                    epsilon2=epsilon2,
                    epsilon3=epsilon3
                )
                future_mel = window_mel + weighted_perturbed
                
                # 다시 파형으로 변환하고, 마지막 부분을 보호된 현재 청크로 사용
                future_wave = self.converter.mel_to_waveform(future_mel.squeeze(0))
                protected_chunk = future_wave[:, -chunk_tensor.shape[1]:]
            
            # 출력 스트림에 쓰기
            output_stream.write(protected_chunk.cpu().numpy())
    
    def _protect_waveform(self, waveform: torch.Tensor, window_size: int = 78,
                         future_step: int = 25, epsilon1: float = 0.115,
                         epsilon2: float = 0.085, epsilon3: float = 0.1) -> torch.Tensor:
        """
        파형 데이터 보호
        
        Args:
            waveform: 입력 파형 [1, T]
            window_size: 슬라이딩 윈도우 크기
            future_step: 예측할 미래 시간 스텝 수
            epsilon1: 저주파 교란의 최대 크기
            epsilon2: 중주파 교란의 최대 크기
            epsilon3: 고주파 교란의 최대 크기
            
        Returns:
            보호된 파형 [1, T]
        """
        # 멜 스펙트로그램으로 변환
        mel_spec = self.converter.waveform_to_mel(waveform)

        # 수정: waveform_to_mel은 [1, F, T] 형태를 반환하므로, predictive_model 입력에 맞게 [B, 1, F, T]로 확장
        if mel_spec.dim() == 3:
            mel_spec = mel_spec.unsqueeze(0)
        
        # 시작 부분에 범용 교란 헤더 적용
        # 수정: header와 mel_spec의 주파수/시간 shape가 다를 수 있으므로 안전하게 crop
        header_freq = min(mel_spec.shape[2], self.header.header.shape[2])
        header_length = min(mel_spec.shape[-1], self.header.header.shape[-1])

        perturbed_mel = mel_spec.clone()
        perturbed_mel[:, :, :header_freq, :header_length] += \
            self.header.header[:, :, :header_freq, :header_length]
        
        # 슬라이딩 윈도우와 예측 모델을 사용해 실시간 교란 생성
        # 수정: 기존 range(0, mel_spec.shape[-1] - window_size, future_step)는 마지막 구간을 놓칠 수 있으므로 +1 적용
        if mel_spec.shape[-1] > window_size:
            loop_range = range(0, mel_spec.shape[-1] - window_size + 1, future_step)
        else:
            loop_range = []

        for start_idx in tqdm.tqdm(loop_range):
            # 현재 윈도우 가져오기
            window = mel_spec[:, :, :, start_idx:start_idx+window_size]
            
            # 교란 예측
            with torch.no_grad():
                perturbation = self.predictive_model(window)

            # 수정: 예측 교란 shape를 mel_spec에 적용 가능한 범위로 crop
            perturbation = crop_perturbation_to_mel(perturbation, mel_spec)
            
            # 미래 시간 스텝에 교란 적용
            future_idx = start_idx + window_size
            future_end = min(future_idx + perturbation.shape[-1], perturbed_mel.shape[-1])
            
            if future_idx < perturbed_mel.shape[-1]:
                # 수정: 주파수 차원도 공통 범위만 사용
                freq_len = min(perturbed_mel.shape[2], perturbation.shape[2])
                apply_time_len = future_end - future_idx

                if apply_time_len > 0:
                    perturbed_mel[:, :, :freq_len, future_idx:future_end] += \
                        perturbation[:, :, :freq_len, :apply_time_len]
        
        # 가중치 기반 제약 적용
        # 수정: 기존 converter.apply_weighted_constraint()는 3D 입력 기준이므로 4D용 함수 사용
        weighted_perturbed = apply_weighted_constraint_4d(
            perturbed_mel - mel_spec,
            epsilon1=epsilon1,
            epsilon2=epsilon2,
            epsilon3=epsilon3
        )
        perturbed_mel = mel_spec + weighted_perturbed
        
        # 다시 파형으로 변환
        # 수정: mel_to_waveform은 [1, F, T] 입력을 기대하므로 batch 차원 제거
        protected_waveform = self.converter.mel_to_waveform(perturbed_mel.squeeze(0))
        
        return protected_waveform

    @staticmethod
    def _match_waveform_length(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        수정: Griffin-Lim 기반 mel_to_waveform 복원 결과가 원본 길이와 다를 수 있으므로,
        저장 전 원본 waveform 길이에 맞춘다.

        Args:
            waveform: 복원된 waveform [1, T]
            target_length: 맞출 목표 길이

        Returns:
            길이가 target_length로 맞춰진 waveform [1, target_length]
        """
        current_length = waveform.shape[-1]

        if current_length > target_length:
            return waveform[:, :target_length]

        if current_length < target_length:
            pad_len = target_length - current_length
            return torch.nn.functional.pad(waveform, (0, pad_len))

        return waveform


def main():
    parser = argparse.ArgumentParser(description='VSMask: 음성 합성 방어 시스템')
    
    # 모델 인자
    parser.add_argument('--predictive_model', type=str, required=True,
                        help='예측 모델 경로')
    parser.add_argument('--header', type=str, required=True,
                        help='범용 교란 헤더 경로')
    
    # 오디오 파일 인자
    parser.add_argument('--input', type=str, required=True,
                        help='입력 오디오 파일 경로')
    parser.add_argument('--output', type=str, required=True,
                        help='출력 오디오 파일 경로')

    # 수정: train_header.py / train_predictive.py와 동일한 mel 설정을 inference에서도 지정 가능하게 추가
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='오디오 샘플링 레이트')
    parser.add_argument('--n_fft', type=int, default=1024,
                        help='FFT 크기')
    parser.add_argument('--hop_length', type=int, default=256,
                        help='프레임 이동 간격')
    parser.add_argument('--n_mels', type=int, default=80,
                        help='멜 필터뱅크 개수')
    
    # 보호 인자
    # 수정: window_size는 waveform sample이 아니라 mel frame 기준이다.
    #       VSMask 논문 t=1.25초, hop_length=256, sample_rate=16000 기준 약 78 frames.
    parser.add_argument('--window_size', type=int, default=78,
                        help='슬라이딩 윈도우 크기. mel frame 단위. 기본값 78은 16kHz/hop=256 기준 약 1.25초')

    # 수정: future_step은 mel frame 기준이다.
    #       VSMask 논문 Δt 또는 γ=0.4초, hop_length=256, sample_rate=16000 기준 약 25 frames.
    parser.add_argument('--future_step', type=int, default=25,
                        help='예측할 미래 시간 스텝 수. mel frame 단위. 기본값 25는 16kHz/hop=256 기준 약 0.4초')

    # 수정: VSMask 논문 설정 ε=0.10, ε1=1.15ε, ε2=0.85ε, ε3=ε에 맞춤
    parser.add_argument('--epsilon1', type=float, default=0.115,
                        help='저주파 교란의 최대 크기')
    parser.add_argument('--epsilon2', type=float, default=0.085,
                        help='중주파 교란의 최대 크기')
    parser.add_argument('--epsilon3', type=float, default=0.1,
                        help='고주파 교란의 최대 크기')
    
    # 기타 인자
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='사용할 장치')
    
    args = parser.parse_args()

    # 수정: CUDA가 없는 환경에서 --device cuda가 들어오면 자동으로 CPU로 전환
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA를 사용할 수 없습니다. device를 cpu로 변경합니다.")
        args.device = 'cpu'
    
    # VSMask 시스템 초기화
    vsmask = VSMask(
        args.predictive_model,
        args.header,
        device=args.device,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels
    )
    
    # 오디오 파일 보호
    vsmask.protect_file(
        args.input,
        args.output,
        window_size=args.window_size,
        future_step=args.future_step,
        epsilon1=args.epsilon1,
        epsilon2=args.epsilon2,
        epsilon3=args.epsilon3
    )


if __name__ == '__main__':
    main()