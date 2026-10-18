"""답 만들 때 다음 토큰 고르기 (ai_server.py, ai_local_test.py, compare_models.py가 같이 씀).

- 첫 토큰 빈칸('▁') 막기: 답 첫머리에 빈칸이 오면 뒤에 번호 목록 숫자("1. 2. ...")가 잘 따라 나온다.
- 같은 묶음 반복 막기(no_repeat_ngram): 이번 답에 이미 나온 n조각 묶음을 다시 못 쓰게 한다.
  조사·어미 하나("는", "요", ".")는 반복돼도 되고, "'안녕'이라는 단어" 같은 덩어리 반복만 막힌다.
- 반복 벌점(repetition): 이번 답에 쓴 토큰 점수를 깎는다. 조사·어미까지 깎여서 세게 쓰면 문장이 망가진다 (1.0이면 끔).
벌점·막기는 모두 지금 만드는 답에만 적용한다 (사용자 말과 끝 표시는 건드리지 않음).
"""
import torch
import torch.nn.functional as F


def pick_next(logits, step, new, space_id, temperature=0.7, top_k=40, repetition=1.0, no_repeat_ngram=3):
    """logits: [1, vocab] (마지막 위치), new: 이번 답에서 지금까지 고른 토큰들. 고른 토큰 [1, 1]을 돌려준다."""
    logits = logits.clone()
    if step == 0 and space_id is not None:
        logits[:, space_id] = float("-inf")
    n = no_repeat_ngram
    if n and n > 1 and len(new) >= n - 1:
        prefix = tuple(new[len(new) - (n - 1):])
        banned = {new[i + n - 1] for i in range(len(new) - n + 1) if tuple(new[i:i + n - 1]) == prefix}
        if banned:
            logits[0, list(banned)] = float("-inf")
    if repetition > 1.0 and new:
        used = torch.tensor(sorted(set(new) - {space_id}), dtype=torch.long)
        if len(used):
            v = logits[0, used]
            logits[0, used] = torch.where(v > 0, v / repetition, v * repetition)
    if temperature <= 0.01:
        return torch.argmax(logits, dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < values[:, [-1]]] = float("-inf")
    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)


def space_id_of(sp):
    sid = sp.piece_to_id("▁")
    return None if sid == sp.unk_id() else sid
