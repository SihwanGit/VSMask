import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.dataset import get_dataloaders
from models.header_model import UniversalPerturbationHeader
from utils.audio import MelSpectrogramConverter
import tqdm


def move_converter_to_device(converter: MelSpectrogramConverter, device: str) -> MelSpectrogramConverter:
    """
    수정: MelSpectrogramConverter 내부 torchaudio transform들을 현재 device로 이동한다.
    GPU 환경에서 waveform은 cuda에 있는데 mel_transform은 cpu에 남아 있으면 device mismatch가 발생할 수 있다.
    """
    converter.mel_transform = converter.mel_transform.to(device)
    converter.inverse_mel_transform = converter.inverse_mel_transform.to(device)
    converter.griffin_lim = converter.griffin_lim.to(device)
    return converter


def train_universal_header(speaker_encoder, args):
    """
    범용 교란 헤더 학습
    
    Args:
        speaker_encoder: 목표 음성 합성 모델의 화자 인코더
        args: 명령줄 인자
    """
    print("범용 교란 헤더 학습을 시작합니다...")

    # 수정: 현재 학습에 사용되는 주요 시간 파라미터를 출력하여 단위 혼동을 줄임
    print("[설정 확인]")
    print(f"  sample_rate   : {args.sample_rate}")
    print(f"  window_size   : {args.window_size} samples ({args.window_size / args.sample_rate:.3f} sec)")
    print(f"  shift_size    : {args.shift_size} samples ({args.shift_size / args.sample_rate:.3f} sec)")
    print(f"  hop_length    : {args.hop_length} samples")
    print(f"  header_length : {args.header_length} mel frames ({args.header_length * args.hop_length / args.sample_rate:.3f} sec)")
    print(f"  device        : {args.device}")
    
    # 데이터 로더 생성
    train_loader, _ = get_dataloaders(
        args.data_dir, args.target_speaker, args.other_speakers,
        batch_size=args.batch_size, window_size=args.window_size,
        shift_size=args.shift_size, sample_rate=args.sample_rate,
        # 수정: 수정된 dataset.py의 num_workers 인자와 연동
        num_workers=args.num_workers
    )
    
    # 오디오 변환기 초기화
    converter = MelSpectrogramConverter(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels
    )

    # 수정: converter 내부 transform을 현재 device로 이동
    converter = move_converter_to_device(converter, args.device)
    
    # 범용 교란 헤더 초기화
    header = UniversalPerturbationHeader(
        mel_bins=args.n_mels,
        time_length=args.header_length,
        device=args.device
    )
    
    # 옵티마이저 설정
    optimizer = optim.Adam([header.header], lr=args.lr)
    
    # 학습 샘플 수집
    source_mels = []
    target_mels = []
    
    print("학습 샘플을 수집합니다...")
    for batch in tqdm.tqdm(train_loader):
        source_waveform = batch['source_waveform'].to(args.device)
        target_waveform = batch['target_waveform'].to(args.device)
        
        # 멜 스펙트로그램으로 변환
        for i in range(source_waveform.size(0)):
            source_mel = converter.waveform_to_mel(source_waveform[i]).unsqueeze(0)
            target_mel = converter.waveform_to_mel(target_waveform[i]).unsqueeze(0)
            
            # header_length 길이의 구간만 사용
            if source_mel.shape[-1] >= args.header_length and target_mel.shape[-1] >= args.header_length:
                source_mels.append(source_mel[:, :, :, :args.header_length])
                target_mels.append(target_mel[:, :, :, :args.header_length])
        
        # 충분한 샘플을 수집하면 중단
        if len(source_mels) >= args.max_samples:
            break

    # 수정: 수집된 샘플이 없는 경우 torch.cat 에러 대신 원인을 설명하는 에러 출력
    if len(source_mels) == 0 or len(target_mels) == 0:
        raise RuntimeError(
            "학습에 사용할 mel 샘플이 수집되지 않았습니다.\n"
            f"현재 header_length={args.header_length} mel frames "
            f"({args.header_length * args.hop_length / args.sample_rate:.3f} sec)입니다.\n"
            "원인 후보:\n"
            "1. wav 길이가 너무 짧음\n"
            "2. window_size가 너무 작아 mel frame 수가 header_length보다 작음\n"
            "3. dataset.py가 생성한 segment가 너무 짧음\n"
            "해결 방법:\n"
            "- smoke test라면 --header_length를 20 또는 50으로 낮추세요.\n"
            "- 논문 기준 실험이라면 --window_size 26400 이상을 사용하세요."
        )
    
    # 샘플을 쌓아 배치로 구성
    source_mels = torch.cat(source_mels[:args.max_samples], dim=0)
    target_mels = torch.cat(target_mels[:args.max_samples], dim=0)

    # 수정: source_mels와 target_mels의 shape 출력
    print(f"source_mels shape: {tuple(source_mels.shape)}")
    print(f"target_mels shape: {tuple(target_mels.shape)}")
    
    print(f"최적화를 시작합니다. 수집된 샘플 수: {source_mels.size(0)}개")
    # 최적화 수행
    header.optimize(
        source_mels, target_mels, speaker_encoder,
        optimizer, num_iterations=args.iterations,
        epsilon=args.epsilon, lambda_param=args.lambda_param
    )
    
    # 범용 교란 헤더 저장
    os.makedirs(args.output_dir, exist_ok=True)

    # 수정: output_name 인자를 통해 fixed header / 실험별 header 파일명을 구분할 수 있게 함
    output_path = os.path.join(args.output_dir, args.output_name)
    header.save(output_path)
    print(f"범용 교란 헤더가 {output_path}에 저장되었습니다.")


def main():
    parser = argparse.ArgumentParser(description='VSMask 범용 교란 헤더 학습')
    
    # 데이터셋 인자
    parser.add_argument('--data_dir', type=str, default='./data/VCTK-Corpus',
                        help='데이터셋 루트 디렉터리')
    parser.add_argument('--target_speaker', type=str, required=True,
                        help='목표 화자 ID. 보호 대상')
    parser.add_argument('--other_speakers', type=str, nargs='+', required=True,
                        help='기타 화자 ID 목록. 오도 목적으로 사용')
    
    # 오디오 인자
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='오디오 샘플링 레이트')
    parser.add_argument('--n_fft', type=int, default=1024,
                        help='FFT 크기')
    parser.add_argument('--hop_length', type=int, default=256,
                        help='프레임 이동 간격')
    parser.add_argument('--n_mels', type=int, default=80,
                        help='멜 필터뱅크 개수')
    
    # 학습 인자
    parser.add_argument('--batch_size', type=int, default=1,
                        help='배치 크기')

    # 수정: 기존 기본값 100 samples는 16kHz 기준 0.00625초라 너무 짧음.
    #       VSMask 논문 기준 header 길이 t + Δt = 1.65초에 맞춰 26400 samples로 변경.
    parser.add_argument('--window_size', type=int, default=26400,
                        help='슬라이딩 윈도우 크기. waveform sample 단위. 기본값 26400은 16kHz 기준 1.65초')

    # 수정: 기존 기본값 50 samples는 너무 짧음.
    #       VSMask 논문의 Δt 또는 γ = 0.4초에 맞춰 6400 samples로 변경.
    parser.add_argument('--shift_size', type=int, default=6400,
                        help='슬라이딩 윈도우 이동 간격. waveform sample 단위. 기본값 6400은 16kHz 기준 0.4초')

    # 수정: header_length는 waveform sample이 아니라 mel frame 단위.
    #       hop_length=256, sample_rate=16000일 때 100 frames ≈ 1.6초로 VSMask header 1.65초와 근접.
    parser.add_argument('--header_length', type=int, default=100,
                        help='범용 교란 헤더의 시간 길이. mel frame 단위')

    parser.add_argument('--iterations', type=int, default=1000,
                        help='최적화 반복 횟수')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='학습률')

    # 수정: VSMask 논문 설정의 perturbation amplitude constraint ε=0.10에 맞춤
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='교란의 최대 크기')

    # 수정: VSMask 논문은 λ=1로 설정했으므로 기본값을 0.5에서 1.0으로 변경
    parser.add_argument('--lambda_param', type=float, default=1.0,
                        help='손실 함수에서 사용하는 균형 파라미터')

    parser.add_argument('--max_samples', type=int, default=1000,
                        help='학습에 사용할 최대 샘플 수')
    
    # 기타 인자
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='사용할 장치')
    parser.add_argument('--output_dir', type=str, default='./output/fixed_header',
                        help='출력 디렉터리')

    # 수정: 저장 파일명을 인자로 지정할 수 있게 추가
    parser.add_argument('--output_name', type=str, default='universal_header.pt',
                        help='저장할 범용 교란 헤더 파일명')

    # 수정: WSL/CPU smoke test 안정성을 위해 num_workers를 인자로 추가하고 기본값을 0으로 설정
    parser.add_argument('--num_workers', type=int, default=0,
                        help='데이터 로딩 worker 수. WSL/CPU 디버깅에서는 0 권장')
    
    args = parser.parse_args()

    # 수정: CUDA가 없는 환경에서 --device cuda가 들어오면 자동으로 CPU로 전환
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA를 사용할 수 없습니다. device를 cpu로 변경합니다.")
        args.device = 'cpu'
    
    # TODO: 여기서는 목표 음성 합성 모델의 화자 인코더를 로드해야 함
    # 아래 코드는 예시이며, 실제 적용 시에는 실제 인코더를 로드해야 함
    class DummySpeakerEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(32, 128)
        
        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    # 화자 인코더 로드 또는 생성
    speaker_encoder = DummySpeakerEncoder().to(args.device)
    
    # 범용 교란 헤더 학습
    train_universal_header(speaker_encoder, args)


if __name__ == '__main__':
    main()