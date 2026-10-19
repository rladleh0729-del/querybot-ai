"""PyTorch 없이 numpy만으로 돌리는 GPT 모델 (DATA_STUDYING_v2.py의 GPTLanguageModel과 같은 계산).

Render 무료(메모리 512MB)에서 PyTorch는 불러오기만 해도 350MB 넘게 먹어서, 모델을 여러 개 올릴 수가 없다.
이 모델은 구조가 단순해서(LayerNorm·어텐션·MLP, bias 없음, 입출력 가중치 공유) numpy로 충분하다.
앞에서 계산한 k, v를 들고 다니는 KV 캐시로 새 토큰 하나만 계산한다.

가중치 파일은 make_cloud_model.py가 만든 npz(반 정밀도)이고, 불러올 때 float32로 바꾼다.
"""
import io
import json
from pathlib import Path

import numpy as np

EPS = 1e-5
_GELU_C = np.float32(np.sqrt(2.0 / np.pi))


def _layer_norm(x, w, b=None):
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    y = (x - mean) / np.sqrt(var + EPS) * w
    return y + b if b is not None else y


def _gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(_GELU_C * (x + np.float32(0.044715) * x * x * x)))


def _softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def read_parts(path: Path) -> bytes:
    """path 그대로 있으면 읽고, 없으면 path.part01, part02 ... 를 이어 붙인다 (GitHub 웹 업로드 한도 때문에 조각냄)."""
    if path.exists():
        return path.read_bytes()
    parts = sorted(path.parent.glob(path.name + ".part*"))
    if not parts:
        raise FileNotFoundError(path)
    return b"".join(p.read_bytes() for p in parts)


class NPModel:
    def __init__(self, npz_path: Path):
        with np.load(io.BytesIO(read_parts(Path(npz_path))), allow_pickle=False) as z:
            meta = json.loads(bytes(z["__meta__"]).decode("utf-8"))
            w = {k: z[k] for k in z.files if k != "__meta__"}
        self.meta = meta
        cfg = meta["model_config"]
        self.n_head = cfg["n_head"]
        self.n_layer = cfg["n_layer"]
        self.block_size = cfg["block_size"]
        f32 = lambda k: np.ascontiguousarray(w.pop(k), dtype=np.float32)  # noqa: E731
        opt = lambda k: f32(k) if k in w else None  # noqa: E731
        self.wte = f32("transformer.wte.weight")          # [vocab, C] — 출력층과 같이 씀
        self.wpe = f32("transformer.wpe.weight")          # [block, C]
        self.layers = []
        for i in range(self.n_layer):
            p = f"transformer.h.{i}."
            self.layers.append({
                "ln1": f32(p + "ln_1.weight"), "ln1b": opt(p + "ln_1.bias"),
                # 곱셈이 빠르게 미리 뒤집어 둔다: x @ W.T
                "qkv": np.ascontiguousarray(f32(p + "attn.qkv_proj.weight").T),
                "out": np.ascontiguousarray(f32(p + "attn.out_proj.weight").T),
                "ln2": f32(p + "ln_2.weight"), "ln2b": opt(p + "ln_2.bias"),
                "fc": np.ascontiguousarray(f32(p + "mlp.fc.weight").T),
                "proj": np.ascontiguousarray(f32(p + "mlp.proj.weight").T),
            })
        self.lnf = f32("transformer.ln_f.weight")
        self.lnfb = opt("transformer.ln_f.bias")
        self.params = int(meta.get("params", 0))

    def __call__(self, ids, past=None):
        """ids: 토큰 번호 목록. past: 앞에서 받은 캐시(처음엔 None). 마지막 위치 logits[vocab]과 새 캐시를 돌려준다."""
        start = 0 if past is None else past[0][0].shape[1]
        t = len(ids)
        if start + t > self.block_size:
            raise ValueError("block_size 초과")
        x = self.wte[ids] + self.wpe[start:start + t]          # [T, C]
        c = x.shape[1]
        hd = c // self.n_head
        scale = np.float32(1.0 / np.sqrt(hd))
        presents = []
        for i, L in enumerate(self.layers):
            h = _layer_norm(x, L["ln1"], L["ln1b"])
            q, k, v = np.split(h @ L["qkv"], 3, axis=1)            # 각 [T, C]
            q = q.reshape(t, self.n_head, hd).transpose(1, 0, 2)   # [H, T, hd]
            k = k.reshape(t, self.n_head, hd).transpose(1, 0, 2)
            v = v.reshape(t, self.n_head, hd).transpose(1, 0, 2)
            if past is not None:
                k = np.concatenate((past[i][0], k), axis=1)
                v = np.concatenate((past[i][1], v), axis=1)
            presents.append((k, v))
            att = (q @ k.transpose(0, 2, 1)) * scale               # [H, T, S]
            if t > 1:  # 처음(질문 전체)은 앞만 보게 가린다
                s = k.shape[1]
                mask = np.triu(np.ones((t, s), dtype=bool), k=s - t + 1)
                att = np.where(mask, np.float32(-np.inf), att)
            y = (_softmax(att) @ v).transpose(1, 0, 2).reshape(t, c)
            x = x + y @ L["out"]
            h = _layer_norm(x, L["ln2"], L["ln2b"])
            x = x + _gelu_tanh(h @ L["fc"]) @ L["proj"]
        last = _layer_norm(x[-1], self.lnf, self.lnfb)
        return last @ self.wte.T, presents
