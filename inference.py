import argparse

import soundfile as sf
import torch

from data_utils import denormalize, file2mel, load_model, mel2wav, normalize


def main(model_dir: str, source: str, target: str, output: str):
    """음성 변환 수행
    
    모델을 로드하고 음성 변환을 수행한 뒤, 결과를 지정된 경로에 저장한다.
    
    Args:
        model_dir: 모델 파일 디렉터리
        source: 원본 음성 파일 경로. 언어 내용을 제공함
        target: 목표 음성 파일 경로. 음색 특징을 제공함
        output: 출력 파일 경로
    """
    # 모델과 설정 로드
    model, config, attr, device = load_model(model_dir)

    # 오디오 파일을 멜 스펙트로그램으로 변환
    src_mel = file2mel(source, **config["preprocess"])
    tgt_mel = file2mel(target, **config["preprocess"])

    # 정규화
    src_mel = normalize(src_mel, attr)
    tgt_mel = normalize(tgt_mel, attr)

    # 텐서로 변환한 뒤 장치로 이동
    src_mel = torch.from_numpy(src_mel).T.unsqueeze(0).to(device)
    tgt_mel = torch.from_numpy(tgt_mel).T.unsqueeze(0).to(device)

    # 음성 변환 수행
    with torch.no_grad():
        out_mel = model.inference(src_mel, tgt_mel)
        out_mel = out_mel.squeeze(0).T
    
    # 역정규화
    out_mel = denormalize(out_mel.data.cpu().numpy(), attr)
    
    # 다시 파형으로 변환
    out_wav = mel2wav(out_mel, **config["preprocess"])

    # 결과 저장
    sf.write(output, out_wav, config["preprocess"]["sample_rate"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str, help="The directory of model files.")
    parser.add_argument(
        "source", type=str, help="The source utterance providing linguistic content."
    )
    parser.add_argument(
        "target", type=str, help="The target utterance providing vocal timbre."
    )
    parser.add_argument("output", type=str, help="The output converted utterance.")
    main(**vars(parser.parse_args()))