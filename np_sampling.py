"""다음 토큰 고르기 — ai_sampling.py(PyTorch판)를 numpy로 옮긴 것. 규칙은 똑같다.

- 첫 토큰 빈칸('▁') 막기: 답 첫머리에 빈칸이 오면 뒤에 번호 목록 숫자("1. 2. ...")가 잘 따라 나온다.
- 막을 토큰(banned): 사용자 말에 숫자가 없으면 숫자 토큰, 그리고 깨진 글자(�)를 만드는 바이트 조각은 항상 막는다.
- 같은 묶음 반복 막기(no_repeat_ngram): 이번 답에 이미 나온 n조각 묶음을 다시 못 쓰게 한다.
- 반복 벌점(repetition): 이번 답에 쓴 토큰 점수를 깎는다 (1.0이면 끔, 세면 조사·어미가 망가짐).
벌점·막기는 모두 지금 만드는 답에만 적용한다 (사용자 말과 끝 표시는 건드리지 않음).
"""
import re

import numpy as np

_DIGIT = re.compile(r"[0-9]")
_rng = np.random.default_rng()


def pick_next(logits, step, new, space_id, temperature=0.7, top_k=40, repetition=1.0, no_repeat_ngram=3, banned=None):
    """logits: [vocab], new: 이번 답에서 지금까지 고른 토큰들. 고른 토큰 번호(int)를 돌려준다."""
    logits = logits.astype(np.float64, copy=True)
    if step == 0 and space_id is not None:
        logits[space_id] = -np.inf
    if banned is not None and len(banned):
        logits[banned] = -np.inf
    n = no_repeat_ngram
    if n and n > 1 and len(new) >= n - 1:
        prefix = tuple(new[len(new) - (n - 1):])
        banned_ng = {new[i + n - 1] for i in range(len(new) - n + 1) if tuple(new[i:i + n - 1]) == prefix}
        if banned_ng:
            logits[list(banned_ng)] = -np.inf
    if repetition > 1.0 and new:
        used = np.array(sorted(set(new) - {space_id}), dtype=np.int64)
        if len(used):
            v = logits[used]
            logits[used] = np.where(v > 0, v / repetition, v * repetition)
    if temperature <= 0.01:
        return int(np.argmax(logits))
    logits = logits / temperature
    if top_k > 0 and top_k < len(logits):
        kth = np.partition(logits, -top_k)[-top_k]
        logits[logits < kth] = -np.inf
    p = np.exp(logits - logits.max())
    p /= p.sum()
    return int(_rng.choice(len(p), p=p))


def space_id_of(sp):
    sid = sp.piece_to_id("▁")
    return None if sid == sp.unk_id() else sid


_ban_cache: dict = {}


def banned_ids(sp, user_text):
    """사용자 말에 숫자가 없으면 숫자 토큰 + 바이트 조각, 있으면 바이트 조각만 막는다."""
    key = id(sp)
    if key not in _ban_cache:
        digits = [i for i in range(sp.get_piece_size()) if _DIGIT.search(sp.id_to_piece(i))]
        bytes_ = [i for i in range(sp.get_piece_size()) if sp.is_byte(i)]
        _ban_cache[key] = (np.array(sorted(set(digits + bytes_)), dtype=np.int64),
                           np.array(sorted(bytes_), dtype=np.int64))
    with_digits, bytes_only = _ban_cache[key]
    return bytes_only if _DIGIT.search(user_text) else with_digits
