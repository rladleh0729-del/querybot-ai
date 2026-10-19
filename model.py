"""DATA_STUDYING_v2.py의 GPTLanguageModel에서 답 만들기에 필요한 부분만 떼어 낸 것.

학습기 파일(5만 자, numpy·학습 경로 설정 포함)을 통째로 올리지 않으려고 따로 둔다.
가중치 이름은 학습기와 똑같아서 학습기 체크포인트를 그대로 불러올 수 있다.

학습기와 다른 점: KV 캐시. 학습기는 토큰 하나 만들 때마다 앞 문장 전체를 다시 계산하는데,
여기서는 앞에서 계산한 값(k, v)을 들고 다녀서 새 토큰 하나만 계산한다. 클라우드 무료 CPU처럼
느린 곳에서 답 시간이 크게 줄어든다. 결과(확률)는 학습기 방식과 같다.
"""
import io
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 256
    n_embd: int = 512
    n_head: int = 8
    n_layer: int = 8
    dropout: float = 0.1
    bias: bool = False


def make_layer_norm(dim: int, bias: bool) -> nn.Module:
    try:
        return nn.LayerNorm(dim, bias=bias)
    except TypeError:
        return nn.LayerNorm(dim)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x, past=None):
        batch_size, seq_len, channels = x.shape
        q, k, v = self.qkv_proj(x).split(channels, dim=2)
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        if past is not None:
            k = torch.cat((past[0], k), dim=2)
            v = torch.cat((past[1], v), dim=2)
        # 처음(질문 전체)은 앞만 보게 가리고, 그 뒤 한 토큰씩은 앞 전부를 봐도 된다.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=past is None and seq_len > 1)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.out_proj(y), (k, v)


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = make_layer_norm(config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = make_layer_norm(config.n_embd, config.bias)
        self.mlp = MLP(config)

    def forward(self, x, past=None):
        a, present = self.attn(self.ln_1(x), past)
        x = x + a
        x = x + self.mlp(self.ln_2(x))
        return x, present


class GPTLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": make_layer_norm(config.n_embd, config.bias),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer["wte"].weight = self.lm_head.weight

    def forward(self, idx, past=None):
        """idx: [1, 길이]. past: 앞에서 받은 캐시 (처음엔 None). 마지막 위치 logits와 새 캐시를 돌려준다."""
        start = 0 if past is None else past[0][0].size(2)
        seq_len = idx.size(1)
        if start + seq_len > self.config.block_size:
            raise ValueError("block_size 초과")
        positions = torch.arange(start, start + seq_len, dtype=torch.long)
        x = self.transformer["wte"](idx) + self.transformer["wpe"](positions)
        presents = []
        for i, block in enumerate(self.transformer["h"]):
            x, present = block(x, None if past is None else past[i])
            presents.append(present)
        x = self.transformer["ln_f"](x[:, -1:, :])
        return self.lm_head(x)[:, -1, :], presents


def load_model(ckpt_path):
    """학습기 체크포인트(또는 make_cloud_model.py로 줄인 것)를 불러온다. 반 정밀도로 저장돼 있어도 float32로 돌린다.

    model.pt 대신 model.pt.part01, part02 ... 조각만 있으면 이어 붙여서 읽는다
    (GitHub 웹 업로드가 파일당 25MB까지라 조각내서 올린다).
    """
    ckpt_path = Path(ckpt_path)
    parts = sorted(ckpt_path.parent.glob(ckpt_path.name + ".part*"))
    if not ckpt_path.exists() and parts:
        source = io.BytesIO(b"".join(p.read_bytes() for p in parts))
    else:
        source = ckpt_path
    ckpt = torch.load(source, map_location="cpu", weights_only=False)
    del source
    if "quantized_model" in ckpt:
        # make_cloud_model.py --int8 로 PC에서 미리 8비트로 줄여 둔 모델: 그대로 쓴다
        # (서버에서 줄이면 줄이는 동안 float32 사본이 잠깐 생겨 메모리가 튄다).
        model = ckpt.pop("quantized_model")
        model.eval()
        return model, ckpt
    # Render 무료(메모리 512MB)에서 버티려고 메모리를 아낀다:
    # 빈 껍데기(meta)로 모델을 만들고 불러온 가중치를 그대로 끼운다 (무작위 초기값용 메모리를 따로 안 씀).
    with torch.device("meta"):
        model = GPTLanguageModel(ModelConfig(**ckpt["model_config"]))
    # 학습기의 옛 방식(F.scaled_dot_product_attention이 없을 때)에서만 생기는 가림막 버퍼는 필요 없다.
    state = {k: v.float() for k, v in ckpt.pop("model_state_dict").items() if not k.endswith("causal_mask")}
    model.load_state_dict(state, assign=True)
    del state
    model.lm_head.weight = model.transformer["wte"].weight  # 끼우면서 풀린 입력·출력 가중치 공유를 다시 묶는다
    model.eval()
    return model, ckpt
