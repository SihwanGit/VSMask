import torch
import torch.nn as nn
from torch import Tensor
from tqdm import trange


def e2e_attack(
    model: nn.Module,
    vc_src: Tensor,
    vc_tgt: Tensor,
    adv_tgt: Tensor,
    eps: float,
    n_iters,
) -> Tensor:
    """엔드투엔드 공격
    
    음성 변환 모델의 출력 결과를 직접 조작하여 적대적 샘플을 생성한다.
    
    Args:
        model: 음성 변환 모델
        vc_src: 원본 음성 특징. 언어 내용을 제공함
        vc_tgt: 목표 음성 특징. 음색 특징을 제공하며 보호 대상임
        adv_tgt: 공격 목표 음성 특징
        eps: 교란 크기 상한
        n_iters: 최적화 반복 횟수
        
    Returns:
        교란이 추가된 목표 음성 특징
    """
    ptb = torch.zeros_like(vc_tgt).normal_(0, 1).requires_grad_(True)  # 무작위 교란 초기화
    opt = torch.optim.Adam([ptb])  # 옵티마이저
    criterion = nn.MSELoss()  # 손실 함수
    pbar = trange(n_iters)  # 진행률 표시줄

    with torch.no_grad():
        org_out = model.inference(vc_src, vc_tgt)  # 원본 변환 결과
        tgt_out = model.inference(vc_src, adv_tgt)  # 목표 변환 결과

    for _ in pbar:
        adv_inp = vc_tgt + eps * ptb.tanh()  # tanh를 사용해 교란 범위를 [-eps, eps]로 제한
        adv_out = model.inference(vc_src, adv_inp)  # 교란이 추가된 입력으로 추론 수행
        # 손실 함수: 출력을 공격 목표에 가깝게 만들고, 동시에 원본 출력에서는 멀어지게 함
        loss = criterion(adv_out, tgt_out) - 0.1 * criterion(adv_out, org_out)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return vc_tgt + eps * ptb.tanh()  # 교란이 추가된 입력 반환


def emb_attack(
    model: nn.Module, vc_tgt: Tensor, adv_tgt: Tensor, eps: float, n_iters: int
) -> Tensor:
    """임베딩 공격
    
    화자 인코더의 출력 임베딩 벡터를 조작하여 적대적 샘플을 생성한다.
    
    Args:
        model: 음성 변환 모델
        vc_tgt: 목표 음성 특징. 음색 특징을 제공하며 보호 대상임
        adv_tgt: 공격 목표 음성 특징
        eps: 교란 크기 상한
        n_iters: 최적화 반복 횟수
        
    Returns:
        교란이 추가된 목표 음성 특징
    """
    ptb = torch.zeros_like(vc_tgt).normal_(0, 1).requires_grad_(True)  # 무작위 교란 초기화
    opt = torch.optim.Adam([ptb])  # 옵티마이저
    criterion = nn.MSELoss()  # 손실 함수
    pbar = trange(n_iters)  # 진행률 표시줄

    with torch.no_grad():
        org_emb = model.speaker_encoder(vc_tgt)  # 원본 화자 임베딩
        tgt_emb = model.speaker_encoder(adv_tgt)  # 목표 화자 임베딩

    for _ in pbar:
        adv_inp = vc_tgt + eps * ptb.tanh()  # tanh를 사용해 교란 범위를 [-eps, eps]로 제한
        adv_emb = model.speaker_encoder(adv_inp)  # 교란이 추가된 입력으로 화자 임베딩 획득
        # 손실 함수: 화자 임베딩을 공격 목표에 가깝게 만들고, 동시에 원본 임베딩에서는 멀어지게 함
        loss = criterion(adv_emb, tgt_emb) - 0.1 * criterion(adv_emb, org_emb)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return vc_tgt + eps * ptb.tanh()  # 교란이 추가된 입력 반환


def fb_attack(
    model: nn.Module,
    vc_src: Tensor,
    vc_tgt: Tensor,
    adv_tgt: Tensor,
    eps: float,
    n_iters: int,
) -> Tensor:
    """피드백 공격
    
    변환된 음성의 화자 임베딩을 조작하여 적대적 샘플을 생성한다.
    
    Args:
        model: 음성 변환 모델
        vc_src: 원본 음성 특징. 언어 내용을 제공함
        vc_tgt: 목표 음성 특징. 음색 특징을 제공하며 보호 대상임
        adv_tgt: 공격 목표 음성 특징
        eps: 교란 크기 상한
        n_iters: 최적화 반복 횟수
        
    Returns:
        교란이 추가된 목표 음성 특징
    """
    ptb = torch.zeros_like(vc_tgt).normal_(0, 1).requires_grad_(True)  # 무작위 교란 초기화
    opt = torch.optim.Adam([ptb])  # 옵티마이저
    criterion = nn.MSELoss()  # 손실 함수
    pbar = trange(n_iters)  # 진행률 표시줄

    with torch.no_grad():
        org_emb = model.speaker_encoder(model.inference(vc_src, vc_tgt))  # 원본 변환 결과의 화자 임베딩
        tgt_emb = model.speaker_encoder(adv_tgt)  # 목표 화자 임베딩

    for _ in pbar:
        adv_inp = vc_tgt + eps * ptb.tanh()  # tanh를 사용해 교란 범위를 [-eps, eps]로 제한
        adv_emb = model.speaker_encoder(model.inference(vc_src, adv_inp))  # 교란이 추가된 입력으로 추론한 뒤 화자 임베딩 획득
        # 손실 함수: 변환 결과의 화자 임베딩을 공격 목표에 가깝게 만들고, 동시에 원본 임베딩에서는 멀어지게 함
        loss = criterion(adv_emb, tgt_emb) - 0.1 * criterion(adv_emb, org_emb)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return vc_tgt + eps * ptb.tanh()  # 교란이 추가된 입력 반환