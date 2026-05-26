import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm

from data.dataset import get_dataloaders
from models.predictive_model import PredictiveModel
from models.header_model import UniversalPerturbationHeader
from utils.audio import MelSpectrogramConverter

"""
핵심 수정 사항
- PredictiveModel 출력 shape가 source_mels와 다를 때 crop해서 맞춤.
- converter.apply_weighted_constraint()가 3차원 입력만 받는 문제를 피하기 위해, train_predictive.py 내부에 4차원용 안전 함수를 추가.
- CPU 환경에서 --device cuda를 잘못 넣었을 때 자동으로 CPU로 fallback.
- future_steps가 너무 커서 학습이 한 번도 안 되는 경우 avg_loss 계산 에러가 나지 않도록 방어.
"""

def apply_weighted_constraint_4d(perturbation: torch.Tensor,
                                 epsilon1: float = 0.1,
                                 epsilon2: float = 0.05,
                                 epsilon3: float = 0.08) -> torch.Tensor:
    """
    수정: train_predictive.py에서 사용하는 perturbation은 [B, 1, F, T] 형태의 4차원 텐서다.
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


def crop_perturbation_to_source(predicted_perturbation: torch.Tensor,
                                source_mels: torch.Tensor) -> torch.Tensor:
    """
    수정: PredictiveModel의 출력 주파수/시간 크기가 source_mels와 정확히 일치하지 않을 수 있다.
    baseline 실행을 우선 목표로 하므로, source_mels에 적용 가능한 범위로 crop한다.

    Args:
        predicted_perturbation: 예측된 교란 [B, 1, Fp, Tp]
        source_mels: 원본 멜 스펙트로그램 [B, 1, Fs, Ts]

    Returns:
        source_mels에 적용 가능한 크기로 잘린 교란 [B, 1, min(Fp, Fs), min(Tp, Ts)]
    """
    freq_len = min(source_mels.shape[2], predicted_perturbation.shape[2])
    time_len = min(source_mels.shape[3], predicted_perturbation.shape[3])

    return predicted_perturbation[:, :, :freq_len, :time_len]


def train_predictive_model(speaker_encoder, args):
    """
    예측 모델 학습
    
    Args:
        speaker_encoder: 목표 음성 합성 모델의 화자 인코더
        args: 명령줄 인자
    """
    print("예측 모델 학습을 시작합니다...")
    
    # 데이터 로더 생성
    train_loader, _ = get_dataloaders(
        args.data_dir, args.target_speaker, args.other_speakers,
        batch_size=args.batch_size, window_size=args.window_size,
        shift_size=args.shift_size, sample_rate=args.sample_rate
    )
    
    # 오디오 변환기 초기화
    converter = MelSpectrogramConverter(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels
    )
    
    # 예측 모델 초기화
    model = PredictiveModel(
        mel_bins=args.n_mels,
        time_dim=args.window_size
    ).to(args.device)

    # 수정: 현재 PredictiveModel 내부에서는 time_dim을 실제로 사용하지 않는다.
    #       따라서 args.window_size가 waveform sample 단위여도 모델 생성 자체에는 영향이 없다.
    #       실제 shape 불일치는 forward 이후 crop_perturbation_to_source()에서 방어한다.
    
    # 범용 교란 헤더 로드. 존재하는 경우에만 사용
    header = None
    if args.header_path and os.path.exists(args.header_path):
        header = UniversalPerturbationHeader(
            mel_bins=args.n_mels,
            time_length=args.header_length,
            device=args.device
        )
        header.load(args.header_path)
        print(f"범용 교란 헤더를 로드했습니다: {args.header_path}")
    
    # 옵티마이저와 손실 함수 설정
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 학습 루프
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        # 수정: future_steps가 너무 커서 학습 update가 한 번도 발생하지 않는 경우를 방어하기 위한 카운터
        update_count = 0
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        for batch in tqdm.tqdm(train_loader):
            source_waveform = batch['source_waveform'].to(args.device)
            target_waveform = batch['target_waveform'].to(args.device)
            
            # 멜 스펙트로그램으로 변환
            source_mels = []
            target_mels = []
            
            for i in range(source_waveform.size(0)):
                source_mel = converter.waveform_to_mel(source_waveform[i]).unsqueeze(0)
                target_mel = converter.waveform_to_mel(target_waveform[i]).unsqueeze(0)
                
                # 범용 교란 헤더가 있으면 입력에 적용
                if header:
                    # 앞부분에 적용
                    # 수정: header와 source_mel의 주파수/시간 크기가 다를 수 있으므로 둘 다 안전하게 맞춤
                    header_freq = min(source_mel.shape[2], header.header.shape[2])
                    header_length = min(source_mel.shape[-1], header.header.shape[-1])

                    source_mel[:, :, :header_freq, :header_length] += \
                        header.header[:, :, :header_freq, :header_length]
                
                source_mels.append(source_mel)
                target_mels.append(target_mel)
            
            source_mels = torch.cat(source_mels, dim=0)
            target_mels = torch.cat(target_mels, dim=0)
            
            # 교란 예측
            predicted_perturbation = model(source_mels)

            # 수정: PredictiveModel 출력 shape가 source_mels와 다를 수 있으므로 적용 가능한 크기로 crop
            predicted_perturbation = crop_perturbation_to_source(
                predicted_perturbation,
                source_mels
            )
            
            # 예측된 교란 적용
            future_idx = args.future_steps
            if future_idx < source_mels.shape[-1]:
                # 예측된 교란을 미래 시간 스텝에 적용
                perturbed_mels = source_mels.clone()

                # 수정: predicted_perturbation의 시간 길이를 고려해 실제 적용 가능한 future_end 계산
                future_end = min(
                    future_idx + predicted_perturbation.shape[-1],
                    perturbed_mels.shape[-1]
                )

                # 수정: 주파수 차원도 predicted_perturbation과 source_mels가 겹치는 범위만 사용
                freq_len = min(perturbed_mels.shape[2], predicted_perturbation.shape[2])
                apply_time_len = future_end - future_idx

                if apply_time_len <= 0:
                    continue

                perturbed_mels[:, :, :freq_len, future_idx:future_end] += \
                    predicted_perturbation[:, :, :freq_len, :apply_time_len]
                
                # 가중치 기반 제약 적용
                # 수정: 기존 converter.apply_weighted_constraint()는 3차원 입력을 가정하므로
                #       [B, 1, F, T] 입력을 처리하는 apply_weighted_constraint_4d()를 사용
                weighted_perturbed = apply_weighted_constraint_4d(
                    perturbed_mels - source_mels,
                    epsilon1=args.epsilon1,
                    epsilon2=args.epsilon2,
                    epsilon3=args.epsilon3
                )

                perturbed_mels = source_mels + weighted_perturbed
                
                # 화자 임베딩 획득
                source_embedding = speaker_encoder(source_mels)
                target_embedding = speaker_encoder(target_mels)
                perturbed_embedding = speaker_encoder(perturbed_mels)
                
                # 손실 계산: 목표 화자와의 거리는 최소화하고, 원본 화자와의 거리는 최대화
                loss_target = nn.MSELoss()(perturbed_embedding, target_embedding)
                loss_source = nn.MSELoss()(perturbed_embedding, source_embedding)
                
                loss = loss_target - args.lambda_param * loss_source
                
                # 그래디언트 업데이트
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                update_count += 1
            
        # 평균 손실을 계산하고 학습률 업데이트
        # 수정: update_count가 0이면 0으로 나누는 문제가 생기므로 명확한 에러를 출력
        if update_count == 0:
            raise RuntimeError(
                "이번 epoch에서 학습 update가 한 번도 발생하지 않았습니다.\n"
                f"future_steps={args.future_steps}, source_mels 시간 길이가 너무 짧을 가능성이 큽니다.\n"
                "해결 방법:\n"
                "1. --future_steps 값을 줄이세요. smoke test라면 5 또는 10 권장\n"
                "2. --window_size 값을 키워 source_mels의 시간 길이를 늘리세요\n"
                "3. --hop_length 값을 줄여 mel frame 수를 늘리세요"
            )

        avg_loss = total_loss / update_count
        scheduler.step(avg_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.6f}")
        
        # 모델 저장
        if (epoch + 1) % args.save_interval == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            model_path = os.path.join(args.output_dir, f'predictive_model_epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), model_path)
            print(f"모델이 {model_path}에 저장되었습니다.")
    
    # 최종 모델 저장
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, 'predictive_model_final.pt')
    torch.save(model.state_dict(), model_path)
    print(f"최종 모델이 {model_path}에 저장되었습니다.")


def main():
    parser = argparse.ArgumentParser(description='VSMask 예측 모델 학습')
    
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
    parser.add_argument('--batch_size', type=int, default=32,
                        help='배치 크기')
    parser.add_argument('--window_size', type=int, default=100,
                        help='슬라이딩 윈도우 크기')
    parser.add_argument('--shift_size', type=int, default=50,
                        help='슬라이딩 윈도우 이동 간격')
    parser.add_argument('--header_length', type=int, default=100,
                        help='범용 교란 헤더의 시간 길이')
    parser.add_argument('--future_steps', type=int, default=10,
                        help='예측할 미래 시간 스텝 수')
    parser.add_argument('--epochs', type=int, default=100,
                        help='학습 에폭 수')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='학습률')
    parser.add_argument('--epsilon1', type=float, default=0.1,
                        help='저주파 교란의 최대 크기')
    parser.add_argument('--epsilon2', type=float, default=0.05,
                        help='중주파 교란의 최대 크기')
    parser.add_argument('--epsilon3', type=float, default=0.08,
                        help='고주파 교란의 최대 크기')
    parser.add_argument('--lambda_param', type=float, default=0.5,
                        help='손실 함수에서 사용하는 균형 파라미터')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='모델 저장 에폭 간격')
    
    # 기타 인자
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='사용할 장치')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='출력 디렉터리')
    parser.add_argument('--header_path', type=str, default=None,
                        help='범용 교란 헤더 경로. 존재하는 경우 사용')
    
    args = parser.parse_args()

    # 수정: GPU가 없는 환경에서 --device cuda가 들어오면 자동으로 CPU로 전환
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
    
    # 예측 모델 학습
    train_predictive_model(speaker_encoder, args)


if __name__ == '__main__':
    main()