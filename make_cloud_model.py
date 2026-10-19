"""학습기 체크포인트를 클라우드에 올릴 크기로 줄여서 model_files/에 넣는다.

  python make_cloud_model.py <best_checkpoint.pth> <spm_small_chat_gpt.model> [--prefix 이름]

- --prefix 없이: 기본(최신) 모델 → model.pt.partNN + spm_small_chat_gpt.model
- --prefix stage3: 방문자가 골라 보는 다른 모델 → stage3.model.pt.partNN + stage3.spm.model
  (app.py의 EXTRA_MODELS에 같은 이름이 있어야 목록에 나온다)
- --int8: 8비트로 미리 줄여서 저장 (다른 모델용. 메모리 약 1/3, 답은 거의 같음)

- 옵티마이저 등 서버에 필요 없는 값은 빼고, 가중치는 반 정밀도(float16)로 저장 → 크기 약 절반.
- GitHub 웹 업로드가 파일당 25MB까지라 9MB씩 model.pt.part01, part02 ... 로 자른다.
- GitHub 웹 업로드는 폴더를 못 만들어서, 저장소에는 model_files/ 안의 파일을 맨 위(app.py 옆)에 올린다.
- 서버는 불러올 때 다시 float32로 바꿔서 돌린다.
- 체크포인트에 토크나이저 지문이 있으면 짝이 맞는지 확인한다.
"""
import hashlib
import io
import shutil
import sys
from pathlib import Path

import torch

PART_BYTES = 9 * 1024 * 1024  # GitHub 웹 업로드(25MB)·브라우저 업로드 도구(10MB) 한도 아래로 자른다
KEEP_KEYS = ("iter_num", "best_val_loss", "model_config", "vocab_size", "tokenizer_sha256", "saved_at")


def main() -> None:
    args = sys.argv[1:]
    prefix = ""
    int8 = "--int8" in args
    if int8:
        args.remove("--int8")
    if "--prefix" in args:
        i = args.index("--prefix")
        prefix = args[i + 1]
        del args[i:i + 2]
    if len(args) != 2:
        raise SystemExit(__doc__)
    ckpt_path, spm_path = Path(args[0]), Path(args[1])
    stem = f"{prefix}.model.pt" if prefix else "model.pt"
    spm_name = f"{prefix}.spm.model" if prefix else "spm_small_chat_gpt.model"
    out_dir = Path(__file__).resolve().parent / "model_files"
    out_dir.mkdir(exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    expected = ckpt.get("tokenizer_sha256")
    actual = hashlib.sha256(spm_path.read_bytes()).hexdigest()
    if expected and expected != actual:
        raise SystemExit(f"토크나이저 짝이 안 맞음\n  체크포인트: {expected}\n  파일: {actual}")

    slim = {k: ckpt[k] for k in KEEP_KEYS if k in ckpt}
    slim["model_state_dict"] = {
        k: (v.half() if v.is_floating_point() else v)
        for k, v in ckpt["model_state_dict"].items()
        if not k.endswith("causal_mask")
    }
    if int8:
        import torch.nn as nn
        from model import load_model
        model, _ = load_model(ckpt_path)
        slim = {k: ckpt[k] for k in KEEP_KEYS if k in ckpt}
        slim["params"] = sum(p.numel() for p in model.parameters())
        slim["quantized_model"] = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    buf = io.BytesIO()
    torch.save(slim, buf)
    data = buf.getvalue()
    for old in list(out_dir.glob(stem + "*")):
        old.unlink()
    parts = [data[i:i + PART_BYTES] for i in range(0, len(data), PART_BYTES)]
    for n, chunk in enumerate(parts, 1):
        (out_dir / f"{stem}.part{n:02d}").write_bytes(chunk)
    shutil.copyfile(spm_path, out_dir / spm_name)
    print(f"저장: {out_dir} ({stem} {len(data) / 1e6:.1f}MB → 조각 {len(parts)}개), "
          f"학습 {slim.get('iter_num', 0):,}번째, val {float(slim.get('best_val_loss', 0)):.3f}")


if __name__ == "__main__":
    main()
