import argparse

import soundfile as sf
import torch

from attack_utils import e2e_attack, emb_attack, fb_attack
from data_utils import denormalize, file2mel, load_model, mel2wav, normalize


def main(
    model_dir: str,
    vc_src: str,
    vc_tgt: str,
    adv_tgt: str,
    output: str,
    eps: float,
    n_iters: int,
    attack_type: str,
):
    """적대적 공격 수행
    
    목표 음성에 적대적 공격을 수행하여 음성 변환 과정에서 비정상적으로 동작하도록 만든다.
    
    Args:
        model_dir: 모델 파일 디렉터리
        vc_src: 원본 음성 파일 경로. 언어 내용을 제공하며 e2e 및 fb 공격에서만 사용됨
        vc_tgt: 목표 음성 파일 경로. 보호해야 하는 음성
        adv_tgt: 공격 목표 음성 파일 경로
        output: 출력 파일 경로
        eps: 교란 크기 상한
        n_iters: 최적화 반복 횟수
        attack_type: 공격 유형. 'e2e', 'emb', 또는 'fb'
    """
    # 파라미터 유효성 검사
    assert attack_type == "emb" or vc_src is not None
    
    # 모델과 설정 로드
    model, config, attr, device = load_model(model_dir)

    # 오디오 파일을 멜 스펙트로그램으로 변환
    vc_tgt = file2mel(vc_tgt, **config["preprocess"])
    adv_tgt = file2mel(adv_tgt, **config["preprocess"])

    # 정규화
    vc_tgt = normalize(vc_tgt, attr)
    adv_tgt = normalize(adv_tgt, attr)

    # 텐서로 변환한 뒤 장치로 이동
    vc_tgt = torch.from_numpy(vc_tgt).T.unsqueeze(0).to(device)
    adv_tgt = torch.from_numpy(adv_tgt).T.unsqueeze(0).to(device)

    # 임베딩 공격이 아닌 경우 원본 음성이 필요함
    if attack_type != "emb":
        vc_src = file2mel(vc_src, **config["preprocess"])
        vc_src = normalize(vc_src, attr)
        vc_src = torch.from_numpy(vc_src).T.unsqueeze(0).to(device)

    # 공격 유형에 따라 해당 공격 수행
    if attack_type == "e2e":  # 엔드투엔드 공격
        adv_inp = e2e_attack(model, vc_src, vc_tgt, adv_tgt, eps, n_iters)
    elif attack_type == "emb":  # 임베딩 공격
        adv_inp = emb_attack(model, vc_tgt, adv_tgt, eps, n_iters)
    elif attack_type == "fb":  # 피드백 공격
        adv_inp = fb_attack(model, vc_src, vc_tgt, adv_tgt, eps, n_iters)
    else:
        raise NotImplementedError()

    # 공격 결과 처리
    adv_inp = adv_inp.squeeze(0).T
    adv_inp = denormalize(adv_inp.data.cpu().numpy(), attr)
    adv_inp = mel2wav(adv_inp, **config["preprocess"])

    # 결과 저장
    sf.write(output, adv_inp, config["preprocess"]["sample_rate"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str, help="The directory of model files.")
    parser.add_argument(
        "vc_tgt",
        type=str,
        help="The target utterance to be defended, providing vocal timbre in voice conversion.",
    )
    parser.add_argument(
        "adv_tgt", type=str, help="The target used in adversarial attack."
    )
    parser.add_argument("output", type=str, help="The output defended utterance.")
    parser.add_argument(
        "--vc_src",
        type=str,
        default=None,
        help="The source utterance providing linguistic content in voice conversion (required in end-to-end and feedback attack).",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.1,
        help="The maximum amplitude of the perturbation.",
    )
    parser.add_argument(
        "--n_iters",
        type=int,
        default=1500,
        help="The number of iterations for updating the perturbation.",
    )
    parser.add_argument(
        "--attack_type",
        type=str,
        choices=["e2e", "emb", "fb"],
        default="emb",
        help="The type of adversarial attack to use (end-to-end, embedding, or feedback attack).",
    )
    main(**vars(parser.parse_args()))